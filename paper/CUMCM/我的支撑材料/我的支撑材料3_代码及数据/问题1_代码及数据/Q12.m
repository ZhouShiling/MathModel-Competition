% 定义土地数据文件名
land_filename = '整理后附件1_1.xlsx';

% 读取土地数据，指定读取第1个sheet，保留变量名
land_data = readtable(land_filename, 'Sheet', 1, 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 地块类型的数组
land_types = {'普通大棚', '智慧大棚', '平旱地', '梯田', '山坡地', '水浇地'};

% 初始化数组来存储地块的名称、类型和面积
plot_names = [];
plot_areas = [];
plot_types = [];

% 遍历每种土地类型
for i = 1:length(land_types)
    type = land_types{i};
    % 获取对应土地类型的地块名称和面积
    names = land_data.('地块名称')(strcmp(land_data.('地块类型'), type));
    areas = land_data.('地块面积')(strcmp(land_data.('地块类型'), type));
    
    % 将地块信息添加到总的数组中
    plot_names = [plot_names; names];
    plot_areas = [plot_areas; areas];
    plot_types = [plot_types; repmat({type}, length(names), 1)];
end

% 创建表格存储地块信息
plot_info = table(plot_names, plot_types, plot_areas, 'VariableNames', {'种植地块', '地块类型', '地块面积'});

% 读取2023年种植数据
planting_filename = '整理后附件2_1.xlsx';
planting_data_2023 = readtable(planting_filename, 'Sheet', 1, 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 将地块信息合并到种植数据中
planting_data_2023 = join(planting_data_2023, plot_info, 'Keys', '种植地块');
% 显示2023年种植数据的前10行
disp('2023年种植数据（前10行）：');
disp(planting_data_2023(1:10, :));  

% 读取作物数据
crop_filename = '整理后附件2_2.xlsx';
crop_data = readtable(crop_filename, 'Sheet', 1, 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 从作物数据中提取关键变量
crop_names = crop_data.('作物名称');                  
plot_types = crop_data.('地块类型');                  
planting_seasons = crop_data.('种植季次');            
yield_per_mu = crop_data.('亩产量/斤');               
planting_costs = crop_data.('种植成本/(元/亩)');      
sale_prices = crop_data.('销售单价/(元/斤)');         

% 读取作物-土地适用数据
filename = '整理后附件1_2.xlsx';  
crop_land_data = readtable(filename, 'Sheet', 1, 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 显示作物-土地适用数据的前30行
disp('导入的作物-土地适用数据:');
disp(crop_land_data(1:30, :));  

% 提取作物编号、名称和类型
crop_ids = crop_land_data.('作物编号');      
crop_names = crop_land_data.('作物名称');    
crop_types = crop_land_data.('作物类型');    
crop_suitable_lands = crop_land_data.('种植耕地');  

% 初始化用于存储数据的变量
land_types_all = {};  
seasons_all = {};     
crop_ids_all = [];    
crop_names_all = {};  
crop_types_all = {};  

% 遍历作物-土地适用数据
for i = 1:height(crop_land_data)
    suitable_lands = crop_suitable_lands{i};
    
    % 移除换行符
    suitable_lands = strrep(suitable_lands, '↵', '');
    
    % 如果土地信息为空则跳过
    if isempty(suitable_lands)
        continue;
    end
    
    % 使用正则表达式解析土地类型和季节
    tokens = regexp(suitable_lands, '(?<land_type>\S+)\s(?<season>\S+)', 'names');
    
    % 遍历解析出的每条记录
    for j = 1:length(tokens)
        land_type = tokens(j).land_type;  
        season = tokens(j).season;  
        
        % 分割季节为数组
        seasons = strsplit(season, ' ');
        for k = 1:length(seasons)
            
            % 将作物信息和土地类型、季节信息添加到数组中
            crop_ids_all(end+1, 1) = crop_ids(i);
            crop_names_all{end+1, 1} = crop_names{i};
            crop_types_all{end+1, 1} = crop_types{i};
            
            land_types_all{end+1, 1} = land_type;
            seasons_all{end+1, 1} = strtrim(seasons{k});
        end
    end
end

% 创建结果表
result_table = table(crop_ids_all, crop_names_all, crop_types_all, land_types_all, seasons_all, ...
                     'VariableNames', {'作物编号', '作物名称', '作物类型', '地块类型', '季节'});

% 显示分解后的作物信息、地块类型和季节
disp('分解后的作物信息、地块类型和季节:');
disp(result_table);

% 将结果表格写入Excel文件
writetable(result_table, '整理后的地块和季节.xlsx');

% 定义未来几年的年份范围
years = 2024:2030;  
% 计算年份的数量
num_years = length(years);

% 从plot_info表格中获取种植地块的名称和地块面积
plot_names = plot_info.('种植地块');  
plot_areas = plot_info.('地块面积');  
% 获取种植地块的数量
num_plots = length(plot_names);       
% 获取作物数据的行数
num_crops = height(crop_data);        

% 定义豆科作物的名称
legume_crops = {'黄豆', '豇豆', '芸豆', '红豆', '黑豆', '绿豆', '爬豆', '刀豆'};

% 定义预期销售因子，用于调整作物的预期销售价格
expected_sales_factor = 0.9;  
% 定义最小地块面积，用于筛选适用的作物
min_plot_area = 0.3;  

% 初始化数组，用于存储每个地块上最后种植的作物
last_crop_planted = cell(num_plots, num_years);  
% 初始化数组，用于记录豆科作物上一次种植的年份
last_legume_year = zeros(num_plots, 1);  

% 遍历未来的每一个年份
for year_idx = 1:num_years
    
    % 初始化数组，用于存储每年第一季的种植计划
    year_plan_first_season = zeros(num_plots, num_crops);  
    % 初始化数组，用于存储每年第二季的种植计划
    year_plan_second_season = zeros(num_plots, num_crops); 

    % 遍历每个种植地块
    for plot_idx = 1:num_plots
        % 获取地块名称和面积
        plot_name = plot_names{plot_idx};
        plot_area = plot_areas(plot_idx);
        
        % 获取当前地块类型适用的作物信息
        applicable_crops = result_table(strcmp(result_table.('地块类型'), plot_info.('地块类型'){plot_idx}), :);
        
        % 如果不是第一年，则排除上一年最后种植的作物
        if year_idx > 1
            last_crop = last_crop_planted{plot_idx, year_idx - 1};
            applicable_crops = applicable_crops(~strcmp(applicable_crops.('作物名称'), last_crop), :);
        end
        
        % 如果上一次种植豆科作物的时间满足一定条件，则优先考虑豆科作物
        if (year_idx - last_legume_year(plot_idx)) >= 3
            crop_names = applicable_crops.('作物名称');
            crop_names = crop_names(~cellfun('isempty', crop_names) & cellfun(@ischar, crop_names));  
            
            legume_crops_applicable = applicable_crops(ismember(crop_names, legume_crops), :);
            
            if ~isempty(legume_crops_applicable)
                applicable_crops = legume_crops_applicable;
            end
        end
        
        % 筛选第一季适用的作物
        season_1_crops = applicable_crops(strcmp(applicable_crops.('季节'), '第一季'), :);
        total_planted_area_first_season = 0;
        if ~isempty(season_1_crops)
            % 为每个作物计算最佳种植面积
            for crop_idx = 1:height(season_1_crops)
                % 调用函数find_best_crop来确定最佳作物和预期收入
                [best_crop_idx, best_revenue] = find_best_crop(season_1_crops(crop_idx, :), crop_data, plot_area, expected_sales_factor, min_plot_area);
                
                % 如果计算结果有效，则更新种植计划和最后种植的作物信息
                if all(best_crop_idx > 0) && total_planted_area_first_season < plot_area
                    planting_area = min(plot_area - total_planted_area_first_season, plot_area); 
                    year_plan_first_season(plot_idx, best_crop_idx) = planting_area;
                    total_planted_area_first_season = total_planted_area_first_season + planting_area;
                    last_crop_planted{plot_idx, year_idx} = crop_data.('作物名称'){best_crop_idx};
                    
                    % 如果最佳作物是豆科作物，更新豆科作物的上一次种植年份
                    if ismember({crop_data.('作物名称'){best_crop_idx}}, legume_crops)
                        last_legume_year(plot_idx) = year_idx;
                    end
                end
            end
        end
        
        % 如果第一季没有种植，则随机选择一个作物进行种植
        if total_planted_area_first_season == 0 && ~isempty(season_1_crops)
            random_crop_idx = randi(height(season_1_crops)); 
            year_plan_first_season(plot_idx, random_crop_idx) = plot_area; 
        end
        
        % 筛选第二季适用的作物
        season_2_crops = applicable_crops(strcmp(applicable_crops.('季节'), '第二季'), :);
        total_planted_area_second_season = 0;
        if ~isempty(season_2_crops)
            % 为每个作物计算最佳种植面积
            for crop_idx = 1:height(season_2_crops)
                [best_crop_idx, best_revenue] = find_best_crop(season_2_crops(crop_idx, :), crop_data, plot_area, expected_sales_factor, min_plot_area);
                
                % 如果计算结果有效，则更新种植计划和最后种植的作物信息
                if all(best_crop_idx > 0) && total_planted_area_second_season < plot_area
                    planting_area = min(plot_area - total_planted_area_second_season, plot_area); 
                    year_plan_second_season(plot_idx, best_crop_idx) = planting_area;
                    total_planted_area_second_season = total_planted_area_second_season + planting_area;
                    last_crop_planted{plot_idx, year_idx} = crop_data.('作物名称'){best_crop_idx};
                    
                    % 如果最佳作物是豆科作物，更新豆科作物的上一次种植年份
                    if ismember({crop_data.('作物名称'){best_crop_idx}}, legume_crops)
                        last_legume_year(plot_idx) = year_idx;
                    end
                end
            end
        end
        
        % 如果第二季没有种植，则随机选择一个作物进行种植
        if total_planted_area_second_season == 0 && ~isempty(season_2_crops)
            random_crop_idx = randi(height(season_2_crops)); 
            year_plan_second_season(plot_idx, random_crop_idx) = plot_area; 
        end
    end
    
    % 将每年的种植计划存储在结构体数组中
    yearly_plans{year_idx} = struct('year', years(year_idx), ...
                                    'first_season', year_plan_first_season, ...
                                    'second_season', year_plan_second_season);
end

% 提取作物名称的唯一值，用于后续分析
crop_names_unique = unique(crop_data.('作物名称'), 'stable');  

% 定义每行的目标值，用于调整种植面积以满足特定的比例要求
row_targets = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, ...
               80, 55, 35, 72, 68, 55, 60, 46, 40, 28, 25, 86, 55, 44, 50, 25, 60, 45, 35, 20, 15, 13, 15, 18, 27, 20, 15, 10, 14, 6, 10, 12, 22, 20];

% 定义最大输出列数，用于限制每行输出的作物种类数量
max_columns_to_output = 42;

% 遍历未来的每一个年份
for year_idx = 1:num_years
    
    % 获取当前年份的种植计划结构体
    year_plan = yearly_plans{year_idx};
    
    % 计算第一季种植计划的列数
    num_columns_first_season = size(year_plan.first_season, 2);
    % 计算第二季种植计划的列数
    num_columns_second_season = size(year_plan.second_season, 2);
    % 获取作物名称的唯一值的数量
    num_crop_names = length(crop_names_unique);
    
    % 初始化第一季每个作物的唯一列索引数组
    unique_crop_names_first = cell(1, num_columns_first_season);
    % 循环填充第一季作物的唯一列索引
    for i = 1:num_columns_first_season
        unique_crop_names_first{i} = crop_names_unique{mod(i-1, num_crop_names) + 1};  
    end
    
    % 初始化第二季每个作物的唯一列索引数组
    unique_crop_names_second = cell(1, num_columns_second_season);
    % 循环填充第二季作物的唯一列索引
    for i = 1:num_columns_second_season
        unique_crop_names_second{i} = crop_names_unique{mod(i-1, num_crop_names) + 1};  
    end
    
    % 获取第一季所有作物名称的唯一值
    unique_names_first = unique(unique_crop_names_first, 'stable');  
    % 初始化第一季所有作物的种植面积数组
    first_season_combined_data = zeros(num_plots, length(unique_names_first));
    % 循环填充第一季所有作物的种植面积
    for i = 1:length(unique_names_first)
        % 获取同一作物的所有列索引
        same_crop_cols = strcmp(unique_crop_names_first, unique_names_first{i});
        % 计算同一作物的所有列的种植面积总和
        first_season_combined_data(:, i) = sum(year_plan.first_season(:, same_crop_cols), 2);
    end
    % 将种植面积数据转换为表格格式
    first_season_combined_table = array2table(first_season_combined_data, 'VariableNames', unique_names_first, 'RowNames', plot_names);
    
    % 获取第二季所有作物名称的唯一值
    unique_names_second = unique(unique_crop_names_second, 'stable');  
    % 初始化第二季所有作物的种植面积数组
    second_season_combined_data = zeros(num_plots, length(unique_names_second));
    % 循环填充第二季所有作物的种植面积
    for i = 1:length(unique_names_second)
        % 获取同一作物的所有列索引
        same_crop_cols = strcmp(unique_crop_names_second, unique_names_second{i});
        % 计算同一作物的所有列的种植面积总和
        second_season_combined_data(:, i) = sum(year_plan.second_season(:, same_crop_cols), 2);
    end
    % 将种植面积数据转换为表格格式
    second_season_combined_table = array2table(second_season_combined_data, 'VariableNames', unique_names_second, 'RowNames', plot_names);
    
    % 如果第一季表格的宽度超过最大输出列数，则裁剪至最大列数
    if width(first_season_combined_table) > max_columns_to_output
        first_season_combined_table = first_season_combined_table(:, 1:max_columns_to_output);
    end
    % 如果第二季表格的宽度超过最大输出列数，则裁剪至最大列数
    if width(second_season_combined_table) > max_columns_to_output
        second_season_combined_table = second_season_combined_table(:, 1:max_columns_to_output);
    end
    
    % 循环遍历每一行，根据行目标值调整种植面积
    for row_idx = 1:height(first_season_combined_table)
        % 计算当前行的第一季种植面积总和
        row_sum_first = sum(first_season_combined_table{row_idx, :});  
        % 如果总和超过目标值，则计算缩放因子并调整面积
        if row_sum_first > row_targets(row_idx)
            scale_factor = row_targets(row_idx) / row_sum_first;  
            first_season_combined_table{row_idx, :} = first_season_combined_table{row_idx, :} * scale_factor;  
        end
        
        % 计算当前行的第二季种植面积总和
        row_sum_second = sum(second_season_combined_table{row_idx, :});
        % 如果总和超过目标值，则计算缩放因子并调整面积
        if row_sum_second > row_targets(row_idx)
            scale_factor = row_targets(row_idx) / row_sum_second;
            second_season_combined_table{row_idx, :} = second_season_combined_table{row_idx, :} * scale_factor;
        end
    end
    
    % 定义第一季种植方案的文件名
    first_season_filename = sprintf('最优种植方案1_第%d季_%d年.xlsx', 1, years(year_idx));
    % 定义第二季种植方案的文件名
    second_season_filename = sprintf('最优种植方案1_第%d季_%d年.xlsx', 2, years(year_idx));
    
    % 将调整后的第一季种植方案写入Excel文件
    writetable(first_season_combined_table, first_season_filename, 'WriteRowNames', true);
    % 将调整后的第二季种植方案写入Excel文件
    writetable(second_season_combined_table, second_season_filename, 'WriteRowNames', true);
end

% 输出消息，表示所有年度的种植方案已成功导出
disp('所有年度的种植方案已成功导出。');