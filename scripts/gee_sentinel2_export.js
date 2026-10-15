/**
 * Google Earth Engine 脚本：梅家坞茶园地块级遥感胁迫信号导出
 * ------------------------------------------------------------------
 * 用途：把 Sentinel-2 的 NDVI / NDRE / NDWI 转成"地块级时间序列"，
 *       供 src/risk_model.py 的 load_rs_indices() 直接读取
 *       （输出列必须为 lat, lon, date, ndvi, ndre, ndwi）。
 *
 * 使用步骤
 * 1. 用 Google 账号登录 https://code.earthengine.google.com/ ，注册一个免费的非商业项目；
 * 2. 把本脚本粘贴到 Code Editor，按需修改 AOI / 日期区间；
 * 3. Run → Tasks 面板中启动导出，CSV 会自动存到你的 Google Drive；
 * 4. 下载后另存为 data/processed/rs_indices.csv。
 *
 * 说明：本脚本同时给出"茶园边界提取"（随机森林分类）与"胁迫指数导出"两部分，
 *       前者用于替换当前用地形适宜性聚类生成的候选管理单元。
 * 合规提示：本脚本只导出植被指数数值，不含任何行政边界或地图渲染内容。
 */

// ------------------------- 1. 研究区与时间区间 -------------------------
var AOI = ee.Geometry.Rectangle([120.0860, 30.1720, 120.1120, 30.2000]); // lonMin, latMin, lonMax, latMax
var START = '2022-06-01';
var END   = '2022-09-15';

// ------------------------- 2. Sentinel-2 预处理 -------------------------
function maskS2(img) {
  var scl = img.select('SCL');
  // 排除：3 云阴影 / 8 中概率云 / 9 高概率云 / 10 卷云 / 11 雪
  var good = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(good).divide(10000)
            .copyProperties(img, ['system:time_start']);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(START, END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2);

print('Sentinel-2 影像数量', s2.size());
Map.centerObject(AOI, 15);

// ------------------------- 3. 植被指数 -------------------------
// NDVI 冠层活力；NDRE 叶绿素（含红边波段，对早期胁迫更敏感）；NDWI 冠层水分
function addIndices(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi');
  var ndre = img.normalizedDifference(['B8', 'B5']).rename('ndre');
  var ndwi = img.normalizedDifference(['B8', 'B11']).rename('ndwi');
  // 冠层含水量：B8/B11 比值，常与水分亏缺相关
  var cmc  = img.select('B8').divide(img.select('B11')).rename('cmc');
  return img.addBands([ndvi, ndre, ndwi, cmc]);
}

var withIdx = s2.map(addIndices);

// ------------------------- 4. 地形因子（供茶园提取与分类共用） -------------------------
var dem  = ee.Image('NASA/NASADEM_HGT/001');
var slope = ee.Terrain.slope(dem);
var aspect = ee.Terrain.aspect(dem);
// 纹理特征：NDVI 的 3×3 标准差，反映成龄茶园的整齐冠层结构
var texture = withIdx.select('ndvi').mean()
  .reduceNeighborhood({reducer: ee.Reducer.stdDev(),
                       kernel: ee.Kernel.square(1.5, 'pixels')})
  .rename('texture');

// ------------------------- 5. 茶园边界提取（可选） -------------------------
// 训练样本：需要人工在 Code Editor 上用"点绘制工具"打标签。
// 做法：新建两个 FeatureCollection 变量 teaSamples / otherSamples，各约 60–100 个点。
// 若无样本，本段可跳过，直接使用下方的指数导出。

var useTeaClassifier = false;   // ← 打好样本后改为 true

var teaMask = null;
if (useTeaClassifier) {
  var s2median = withIdx.median().select(['B2','B3','B4','B8','B11','B12'])
    .addBands(slope).addBands(aspect).addBands(texture).addBands(withIdx.mean().select('ndvi'));
  var training = teaSamples.map(function (f) { return f.set('class', 1); })
    .merge(otherSamples.map(function (f) { return f.set('class', 0); }));
  var classifier = ee.Classifier.smileRandomForest(120).train({
    features: s2median,
    classProperty: 'class',
    inputProperties: s2median.bandNames(),
    // 导出每棵树以评估精度 / 变量重要性
  });
  teaMask = s2median.classify(classifier).eq(1).rename('tea');
  print('变量重要性', classifier.explain());
}

// ------------------------- 6. 地块级指数导出 -------------------------
// 采样点来源：outputs/plot_units_centroids.csv（由 run_pipeline.py 生成）
// 若已获得真实地块边界，改用 ee.Image.reduceRegions(vector, ee.Reducer.mean(), 10) 逐地块统计。
var UNIT_CSV = 'users/your-account/plot_units_centroids';   // ← 上传后替换
var units = ee.FeatureCollection('projects/ee-your-project/assets/plot_units_centroids'); // ← 替换
if (useTeaClassifier && teaMask !== null) {
  // 只保留落在茶园掩膜内的采样点
  units = units.filter(ee.Filter.eq('in_tea', 1));
}

var idxStack = withIdx.select(['ndvi', 'ndre', 'ndwi', 'cmc']);
var sampled = idxStack.addBands(ee.Image().byte().paint(units, 'unit_index').rename('unit_index'))
  .sampleRegions({
    collection: units,
    scale: 10,
    geometries: true,
    tileScale: 4
  });

// 每次影像单独采样会过慢，改为按 10 天合成后统一采样
var ts = withIdx.select(['ndvi', 'ndre', 'ndwi', 'cmc']).map(function (img) {
  return img.set('date', ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'));
});

function sampleByDate(d) {
  var d0 = ee.Date(d);
  var comp = withIdx.select(['ndvi','ndre','ndwi','cmc'])
    .filterDate(d0, d0.advance(10, 'day')).mean();
  var s = comp.sampleRegions({collection: units, scale: 10, geometries: false, tileScale: 4});
  return s.map(function (f) { return f.set('date', d0.format('YYYY-MM-dd')); });
}

var composites = ee.List.sequence(0, ee.Date(END).difference(ee.Date(START), 'day').divide(10).floor().subtract(1))
  .map(function (i) { return sampleByDate(ee.Date(START).advance(ee.Number(i).multiply(10), 'day')); })
  .flatten();

Export.table.toDrive({
  collection: ee.FeatureCollection(composites),
  description: 'meijiawu_rs_indices',
  folder: 'meijiawu_tea',
  fileNamePrefix: 'rs_indices',
  fileFormat: 'CSV',
  selectors: ['lat', 'lon', 'date', 'ndvi', 'ndre', 'ndwi', 'cmc']
});

// ------------------------- 7. 结果可视化 -------------------------
var vis = {min: 0.2, max: 0.9, palette: ['brown', 'yellow', 'green']};
Map.addLayer(withIdx.select('ndvi').mean(), vis, 'NDVI 均值');
Map.addLayer(withIdx.select('ndre').mean(), vis, 'NDRE 均值');
if (teaMask !== null && useTeaClassifier) {
  Map.addLayer(teaMask.selfMask(), {palette: ['00ff88']}, '茶园掩膜（RF 分类）');
}

/* ------------------------------------------------------------------
 * 与主流程对接
 * 1. 导出的 CSV 需包含列 lat, lon, date, ndvi, ndre, ndwi；
 *    （sampleRegions 默认带 .geo 字段，用 QGIS/Excel 从 WKT 提取经纬度即可，
 *      或把 geometries 设为 true 后用 ee.Feature 的 centroid 字段）
 * 2. 保存到 data/processed/rs_indices.csv；
 * 3. 重新运行 python run_pipeline.py —— risk_model.load_rs_indices() 会自动启用
 *    NDVI/NDRE/NDWI 三个特征，特征集从 BASE_FEATURES 扩展到 +RS_FEATURES。
 * ------------------------------------------------------------------ */
