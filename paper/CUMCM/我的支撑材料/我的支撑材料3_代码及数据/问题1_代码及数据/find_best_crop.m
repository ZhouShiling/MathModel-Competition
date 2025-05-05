% 定义函数 find_best_crop，用于找到给定地块和季节条件下，收益最高的作物及相应收益
function [best_crop_idx, best_revenue] = find_best_crop(season_crops, crop_data, plot_area, expected_sales_factor, min_plot_area)

    best_revenue = -Inf;  % 初始化最佳收益为负无穷，用于比较
    best_crop_idx = 0;    % 初始化最佳作物索引为0，表示尚未找到最佳作物

    % 遍历当前季节适用的所有作物
    for crop_idx = 1:height(season_crops)
        crop_name = season_crops.('作物名称'){crop_idx};  % 获取作物名称
        crop_data_idx = find(strcmp(crop_data.('作物名称'), crop_name));  % 在作物数据中查找作物名称的索引

        if ~isempty(crop_data_idx)  % 确保作物数据中存在该作物
            % 从作物数据中提取当前作物的亩产量、销售单价和种植成本
            yield = crop_data.('亩产量/斤')(crop_data_idx);
            sale_price = crop_data.('销售单价/(元/斤)')(crop_data_idx);
            cost = crop_data.('种植成本/(元/亩)')(crop_data_idx);

            % 计算总产量，考虑地块面积
            total_production = yield .* plot_area;  
            % 计算预期销售量，考虑预期销售系数
            expected_sales = expected_sales_factor .* total_production;  

            % 计算收益，考虑实际产量与预期销售量的最小值，减去种植成本
            revenue = min(total_production, expected_sales) .* sale_price - cost .* plot_area;  

            % 如果当前作物的收益大于已找到的最佳收益，并且地块面积大于最小种植面积
            if all(revenue > best_revenue) && all(plot_area >= min_plot_area)
                best_revenue = revenue;  % 更新最佳收益
                best_crop_idx = crop_data_idx;  % 更新最佳作物索引
            end
        end
    end
end