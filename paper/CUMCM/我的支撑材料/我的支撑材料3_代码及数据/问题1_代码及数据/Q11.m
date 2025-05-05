% 定义未来几年的时间范围
years = 2024:2030;  
num_years = length(years);  % 计算年份数量

% 获取所有种植地块的名称和面积
plot_names = plot_info.('种植地块');  
plot_areas = plot_info.('地块面积');  
num_plots = length(plot_names);        % 计算地块数量

% 获取所有作物的数据
num_crops = height(crop_data);         % 计算作物数量

% 定义豆类作物列表
legume_crops = {'黄豆', '豇豆', '芸豆', '红豆', '黑豆', '绿豆', '爬豆', '刀豆'};

% 设置预期的销售系数、最小种植面积
expected_sales_factor = 0.9;  
min_plot_area = 0.3;  

% 初始化数组，记录每个地块每年最后种植的作物和豆类作物最后种植的年份
last_crop_planted = cell(num_plots, num_years);  
last_legume_year = zeros(num_plots, 1);  

% 主循环，遍历每一年
for year_idx = 1:num_years
    
    % 初始化每年第一季和第二季的种植计划矩阵
    year_plan_first_season = zeros(num_plots, num_crops);  
    year_plan_second_season = zeros(num_plots, num_crops); 

    % 遍历每个地块
    for plot_idx = 1:num_plots
        plot_name = plot_names{plot_idx};
        plot_area = plot_areas(plot_idx);

        % 获取当前地块适用的作物列表
        applicable_crops = result_table(strcmp(result_table.('地块类型'), plot_info.('地块类型'){plot_idx}), :);

        % 如果不是第一年，排除上一年在该地块种植的作物
        if year_idx > 1
            last_crop = last_crop_planted{plot_idx, year_idx - 1};
            applicable_crops = applicable_crops(~strcmp(applicable_crops.('作物名称'), last_crop), :);
        end

        % 如果上一次种植豆类作物已经超过3年，可以考虑再次种植豆类
        if (year_idx - last_legume_year(plot_idx)) >= 3
            % 获取当前地块适用的豆类作物列表
            crop_names = applicable_crops.('作物名称');
            % 排除空字符串和非字符类型的作物名称
            crop_names = crop_names(~cellfun('isempty', crop_names) & cellfun(@ischar, crop_names));  
            % 筛选当前地块适用的豆类作物
            legume_crops_applicable = applicable_crops(ismember(crop_names, legume_crops), :);
            % 如果有豆类作物适用，更新适用作物列表为豆类作物
            if ~isempty(legume_crops_applicable)
                applicable_crops = legume_crops_applicable;
            end
        end

        % 筛选第一季适用的作物
        season_1_crops = applicable_crops(strcmp(applicable_crops.('季节'), '第一季'), :);
        total_planted_area_first_season = 0;
        % 如果有第一季作物适用，计算种植面积并更新种植计划
        if ~isempty(season_1_crops)
            for crop_idx = 1:height(season_1_crops)
                % 寻找最佳作物进行种植，考虑预期销售系数和最小种植面积
                [best_crop_idx, best_revenue] = find_best_crop(season_1_crops(crop_idx, :), crop_data, plot_area, expected_sales_factor, min_plot_area);
                % 如果找到最佳作物且种植面积未满，更新种植计划
                if all(best_crop_idx > 0) && total_planted_area_first_season < plot_area
                    planting_area = min(plot_area - total_planted_area_first_season, plot_area); 
                    year_plan_first_season(plot_idx, best_crop_idx) = planting_area;
                    total_planted_area_first_season = total_planted_area_first_season + planting_area;
                    % 记录最后种植的作物
                    last_crop_planted{plot_idx, year_idx} = crop_data.('作物名称'){best_crop_idx};
                    % 如果最佳作物是豆类，更新豆类最后种植年份
                    if ismember({crop_data.('作物名称'){best_crop_idx}}, legume_crops)
                        last_legume_year(plot_idx) = year_idx;
                    end
                end
            end
        end

        % 筛选第二季适用的作物
        season_2_crops = applicable_crops(strcmp(applicable_crops.('季节'), '第二季'), :);
        total_planted_area_second_season = 0;
        % 如果有第二季作物适用，计算种植面积并更新种植计划
        if ~isempty(season_2_crops)
            for crop_idx = 1:height(season_2_crops)
                [best_crop_idx, best_revenue] = find_best_crop(season_2_crops(crop_idx, :), crop_data, plot_area, expected_sales_factor, min_plot_area);
                if all(best_crop_idx > 0) && total_planted_area_second_season < plot_area
                    planting_area = min(plot_area - total_planted_area_second_season, plot_area); 
                    year_plan_second_season(plot_idx, best_crop_idx) = planting_area;
                    total_planted_area_second_season = total_planted_area_second_season + planting_area;
                    last_crop_planted{plot_idx, year_idx} = crop_data.('作物名称'){best_crop_idx};
                    if ismember({crop_data.('作物名称'){best_crop_idx}}, legume_crops)
                        last_legume_year(plot_idx) = year_idx;
                    end
                end
            end
        end

        % 如果第一季和第二季都没有作物种植，随机选择一个作物种植
        if total_planted_area_first_season == 0 && ~isempty(season_1_crops)
            random_crop_idx = randi(height(season_1_crops)); 
            year_plan_first_season(plot_idx, random_crop_idx) = plot_area; 
        end

        if total_planted_area_second_season == 0 && ~isempty(season_2_crops)
            random_crop_idx = randi(height(season_2_crops)); 
            year_plan_second_season(plot_idx, random_crop_idx) = plot_area; 
        end
    end

    % 将每年的种植计划存储在cell数组中
    yearly_plans{year_idx} = struct('year', years(year_idx), ...
                                    'first_season', year_plan_first_season, ...
                                    'second_season', year_plan_second_season);
end

% 获取作物名称的唯一列表
crop_names_unique = unique(crop_data.('作物名称'), 'stable');  

% 定义目标产量或种植比例
row_targets = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, ...
               80, 55, 35, 72, 68, 55, 60, 46, 40, 28, 25, 86, 55, 44, 50, 25, 60, 45, 35, 20, 15, 13, 15, 18, 27, 20, 15, 10, 14, 6, 10, 12, 22, 20];

% 最大输出列数
max_columns_to_output = 42;

% 遍历每一年，处理种植计划
for year_idx = 1:num_years
    
    year_plan = yearly_plans{year_idx};  % 获取当前年份的种植计划

    % 计算第一季和第二季的作物列数
    num_columns_first_season = size(year_plan.first_season, 2);
    num_columns_second_season = size(year_plan.second_season, 2);
    num_crop_names = length(crop_names_unique);  % 获取作物名称数量

    % 生成第一季与第二季的作物名称列表
    unique_crop_names_first = cell(1, num_columns_first_season);
    for i = 1:num_columns_first_season
        unique_crop_names_first{i} = crop_names_unique{mod(i-1, num_crop_names) + 1};  
    end

    unique_crop_names_second = cell(1, num_columns_second_season);
    for i = 1:num_columns_second_season
        unique_crop_names_second{i} = crop_names_unique{mod(i-1, num_crop_names) + 1};  
    end

    % 获取第一季和第二季的独特作物名称（保留原始顺序）
    unique_names_first = unique(unique_crop_names_first, 'stable');  
    unique_names_second = unique(unique_crop_names_second, 'stable');  

    % 合并第一季数据
    first_season_combined_data = zeros(num_plots, length(unique_names_first));
    for i = 1:length(unique_names_first)
        same_crop_cols = strcmp(unique_crop_names_first, unique_names_first{i});  % 找到相同作物的列
        first_season_combined_data(:, i) = sum(year_plan.first_season(:, same_crop_cols), 2);  % 合并数据
    end
    first_season_combined_table = array2table(first_season_combined_data, 'VariableNames', unique_names_first, 'RowNames', plot_names);

    % 合并第二季数据
    second_season_combined_data = zeros(num_plots, length(unique_names_second));
    for i = 1:length(unique_names_second)
        same_crop_cols = strcmp(unique_crop_names_second, unique_names_second{i});  % 找到相同作物的列
        second_season_combined_data(:, i) = sum(year_plan.second_season(:, same_crop_cols), 2);  % 合并数据
    end
    second_season_combined_table = array2table(second_season_combined_data, 'VariableNames', unique_names_second, 'RowNames', plot_names);

    % 限制输出列数，防止超出
    if width(first_season_combined_table) > max_columns_to_output
        first_season_combined_table = first_season_combined_table(:, 1:max_columns_to_output);
    end
    if width(second_season_combined_table) > max_columns_to_output
        second_season_combined_table = second_season_combined_table(:, 1:max_columns_to_output);
    end

    % 调整作物种植量，确保不超出目标
    for row_idx = 1:height(first_season_combined_table)
        row_sum_first = sum(first_season_combined_table{row_idx, :});
        if row_sum_first > row_targets(row_idx)
            scale_factor = row_targets(row_idx) / row_sum_first;  % 调整比例
            first_season_combined_table{row_idx, :} = first_season_combined_table{row_idx, :} * scale_factor;  % 应用调整
        end

        row_sum_second = sum(second_season_combined_table{row_idx, :});
        if row_sum_second > row_targets(row_idx)
            scale_factor = row_targets(row_idx) / row_sum_second;
            second_season_combined_table{row_idx, :} = second_season_combined_table{row_idx, :} * scale_factor;  % 应用调整
        end
    end

    % 保存第一季和第二季的种植计划到Excel文件
    first_season_filename = sprintf('最优种植方案1_第%d年_第一季.xlsx', years(year_idx));
    second_season_filename = sprintf('最优种植方案1_第%d年_第二季.xlsx', years(year_idx));

    % 输出第一季种植数据
    writetable(first_season_combined_table, first_season_filename, 'WriteRowNames', true);

    % 输出第二季种植数据
    writetable(second_season_combined_table, second_season_filename, 'WriteRowNames', true);
end

% 输出完成信息
disp('最优种植方案已成功导出。');