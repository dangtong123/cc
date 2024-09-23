library(terra)
library(sf)
library(data.table)
library(lightgbm)

# 读取网格文件
grid_cells <- st_read("E:/dt/1/guocheng/final_agbd.shp")

# 读取栅格数据
landsat8_files <- list.files("F:/gee/2021/1/", pattern = "^l8_.*\\.tif$", full.names = TRUE)
sentinel2_files <- list.files("F:/gee/2021/1/", pattern = "^s2_.*\\.tif$", full.names = TRUE)
dem_files <- list.files("F:/gee/2021/1/", pattern = "^DEM.*\\.tif$", full.names = TRUE)
land_cover_file <- "F:/gee/2021/1/tdly2021_forest.tif"

# 确保所有栅格与网格对齐
align_rasters <- function(raster_file, target_crs) {
  raster <- rast(raster_file)
  raster <- project(raster, crs(target_crs))
  return(raster)
}

# 对所有栅格进行对齐
landsat8_rasters <- lapply(landsat8_files, align_rasters, target_crs = grid_cells)
sentinel2_rasters <- lapply(sentinel2_files, align_rasters, target_crs = grid_cells)
dem_rasters <- lapply(dem_files, align_rasters, target_crs = grid_cells)

# 计算统计特征的函数
calculate_stats <- function(raster_list, polygons) {
  values_list <- lapply(raster_list, function(raster) {
    values <- terra::extract(raster, polygons, df = TRUE)
    setnames(values, old = "ID", new = "Grid_ID")
    values <- values[complete.cases(values), ]
    band_names <- names(raster)
    stats_list <- list()
    for (band in band_names) {
      stats <- values[, {
        x <- get(band)
        list(
          avg = mean(x, na.rm = TRUE),
          std = sd(x, na.rm = TRUE),
          p2 = quantile(x, 0.02, na.rm = TRUE),
          p25 = quantile(x, 0.25, na.rm = TRUE),
          p50 = quantile(x, 0.5, na.rm = TRUE),
          p75 = quantile(x, 0.75, na.rm = TRUE),
          p98 = quantile(x, 0.98, na.rm = TRUE)
        )
      }, by = Grid_ID]
      
      stat_names <- c("avg", "std", "p2", "p25", "p50", "p75", "p98")
      new_col_names <- paste0(band, "_", stat_names)
      setnames(stats, old = stat_names, new = new_col_names)
      
      stats_list[[band]] <- stats
    }
    stats_dt <- Reduce(function(x, y) merge(x, y, by = "Grid_ID", all = TRUE), stats_list)
    return(stats_dt)
  })
  
  return(values_list)
}

# 计算Landsat 8的统计特征
landsat8_stats <- calculate_stats(landsat8_rasters, grid_cells)

# 计算Sentinel-2的统计特征
sentinel2_stats <- calculate_stats(sentinel2_rasters, grid_cells)

# 计算DEM的统计特征
dem_stats <- calculate_stats(dem_rasters, grid_cells)

# 合并所有自变量统计特征
merge_stats <- function(stats_list) {
  merged_stats <- do.call(Reduce, c(list(function(x, y) merge(x, y, by = "Grid_ID", all = TRUE)), stats_list))
  return(merged_stats)
}

# 合并所有统计数据
predictor_variables <- merge_stats(c(landsat8_stats, sentinel2_stats, dem_stats))

# 合并预测变量与因变量
data_for_model <- merge(grid_cells[, .(ID, avg_agbd)], predictor_variables, by.x = "ID", by.y = "Grid_ID", all = FALSE)
data_for_model <- na.omit(data_for_model)

# 分割数据集
train_index <- sample(nrow(data_for_model), round(0.8 * nrow(data_for_model)))
train_data <- data_for_model[train_index, ]
test_data <- data_for_model[-train_index, ]

# 构建训练数据
train_matrix <- lgb.Dataset(train_data[, -c("ID", "avg_agbd")], label = train_data$avg_agbd)
test_matrix <- lgb.Dataset(test_data[, -c("ID", "avg_agbd")], label = test_data$avg_agbd)

# 设置参数
params <- list(
  objective = "regression",
  metric = "rmse",
  num_leaves = 31,
  learning_rate = 0.05,
  feature_fraction = 0.9,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  verbose = -1
)

# 训练模型
model <- lgb.train(params, train_matrix, num_boost_round = 1000, valid_sets = list(train_matrix, test_matrix), early_stopping_rounds = 100)

# 预测
predictions <- predict(model, test_data[, -c("ID", "avg_agbd")])

# 评估模型性能
rmse <- sqrt(mean((predictions - test_data$avg_agbd)^2))
print(paste("Test RMSE:", rmse))