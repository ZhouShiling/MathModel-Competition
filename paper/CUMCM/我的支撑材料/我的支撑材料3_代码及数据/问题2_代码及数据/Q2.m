% 定义输入文件名
land_filename = '整理后附件1_1.xlsx';

% 读取土地数据表格的第一张sheet，确保变量名读取正确并保留原有命名
land_data = readtable(land_filename, 'Sheet', 1, 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 定义土地类型列表
land_types = {'普通大棚', '智慧大棚', '平旱地', '梯田', '山坡地', '水浇地'};

% 初始化用于存储地块名称、类型和面积的变量
plot_names = [];
plot_areas = [];
plot_types = [];

% 遍历每种土地类型
for i = 1:length(land_types)
    type = land_types{i}; % 获取当前土地类型
    % 筛选出对应类型的地块名称和面积
    names = land_data.('地块名称')(strcmp(land_data.('地块类型'), type));
    areas = land_data.('地块面积')(strcmp(land_data.('地块类型'), type));
    
    % 将找到的地块信息添加到汇总列表
    plot_names = [plot_names; names];
    plot_areas = [plot_areas; areas];
    plot_types = [plot_types; repmat({type}, length(names), 1)];
end

% 将地块信息整合成表格
plot_info = table(plot_names, plot_types, plot_areas, 'VariableNames', {'种植地块', '地块类型', '地块面积'});

% 读取2023年种植数据
planting_filename = '整理后附件2_1.xlsx';
planting_data_2023 = readtable(planting_filename, 'Sheet', 1, 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 将地块信息与种植数据联接
planting_data_2023 = join(planting_data_2023, plot_info, 'Keys', '种植地块');
% 显示前10行的2023年种植数据
disp('2023年种植数据（前10行）：');
disp(planting_data_2023(1:10, :));  

% 读取作物数据
crop_filename = '整理后附件2_2.xlsx';
crop_data = readtable(crop_filename, 'Sheet', 1, 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 从作物数据表中提取关键列
crop_names = crop_data.('作物名称');                  
plot_types = crop_data.('地块类型');                  
planting_seasons = crop_data.('种植季次');            
yield_per_mu = crop_data.('亩产量/斤');               
planting_costs = crop_data.('种植成本/(元/亩)');      
sale_prices = crop_data.('销售单价/(元/斤)');         

% 读取作物-土地适用数据
filename = '整理后附件1_2.xlsx';  
crop_land_data = readtable(filename, 'Sheet', 1, 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 显示读取的作物-土地适用数据的前30行
disp('导入的作物-土地适用数据:');
disp(crop_land_data(1:30, :));  

% 初始化用于存储处理后数据的变量
crop_ids = crop_land_data.('作物编号');      
crop_names = crop_land_data.('作物名称');    
crop_types = crop_land_data.('作物类型');    
% 清洗种植耕地数据，去除换行符
crop_suitable_lands = crop_land_data.('种植耕地');
crop_suitable_lands = strrep(crop_suitable_lands, '↵', '');  

% 初始化用于存储最终结果的变量
land_types_all = {};  
seasons_all = {};     
crop_ids_all = [];    
crop_names_all = {};  
crop_types_all = {};  

% 遍历每条作物数据
for i = 1:height(crop_land_data)
    suitable_lands = crop_suitable_lands{i}; % 获取当前作物的适用土地信息
    
    % 如果土地信息为空，跳过本次循环
    if isempty(suitable_lands)
        continue;  
    end
    
    % 使用正则表达式解析土地类型和季节
    tokens = regexp(suitable_lands, '(?<land_type>\S+)\s(?<season>\S+)', 'names');
    for j = 1:length(tokens)
        land_type = tokens(j).land_type; % 土地类型
        season = tokens(j).season; % 季节
        
        % 将季节字符串按空格分割
        seasons = strsplit(season, ' ');
        for k = 1:length(seasons)
            % 拼接数据到结果列表
            crop_ids_all(end+1, 1) = crop_ids(i);
            crop_names_all{end+1, 1} = crop_names{i};
            crop_types_all{end+1, 1} = crop_types{i};
            land_types_all{end+1, 1} = land_type;
            seasons_all{end+1, 1} = strtrim(seasons{k});
        end
    end
end

% 将数据整合成表格
result_table = table(crop_ids_all, crop_names_all, crop_types_all, land_types_all, seasons_all, ...
                     'VariableNames', {'作物编号', '作物名称', '作物类型', '地块类型', '季节'});

% 显示分解后的作物信息、地块类型和季节
disp('分解后的作物信息、地块类型和季节:');
disp(result_table);

% 将结果保存到新的Excel文件
writetable(result_table, '整理后的地块和季节.xlsx');

% 从作物数据中获取亩产量
initial_yield = crop_data.('亩产量/斤'); 
% 从种植数据中获取地块面积
plot_areas = planting_data_2023.('地块面积'); 

% 获取2023年种植的作物名称
planted_crops = planting_data_2023.('作物名称');

% 初始化每块地的预期亩产量数组
yield_per_plot = zeros(height(planting_data_2023), 1);

% 计算每块地块的预期亩产量
for i = 1:height(planting_data_2023)
    crop_name = planted_crops{i}; % 当前作物名称
    
    % 找到作物在作物数据中的索引
    crop_idx = find(strcmp(crop_data.('作物名称'), crop_name));
    
    % 如果作物在数据中，则获取其亩产量
    if ~isempty(crop_idx)
        yield_per_plot(i) = crop_data.('亩产量/斤')(crop_idx(1));
    else
        yield_per_plot(i) = 0; % 如果作物未找到，亩产量设为0
    end
end

% 计算2023年每个地块的总产量
total_production_2023 = yield_per_plot .* plot_areas;

% 计算2023年每个地块的预期销售量，假定销售量为总产量的80%
initial_sales_volume = total_production_2023 * 0.8;

% 显示每个地块的2023年预期销售量
disp('每个地块的2023年预期销售量:');
disp(initial_sales_volume);

% 重新计算每个作物的初始销售量
initial_sales_volume = zeros(num_crops, 1); % 初始化初始销售量数组

% 计算每个作物的总种植面积和总产量，进而计算初始销售量
for crop_idx = 1:num_crops
    crop_name = crop_data.('作物名称'){crop_idx}; % 当前作物的名称
    yield = crop_data.('亩产量/斤')(crop_idx); % 当前作物的亩产量
    total_area = sum(plot_areas(strcmp(planting_data_2023.('作物名称'), crop_name))); % 计算该作物的总种植面积
    total_production = yield * total_area; % 计算该作物的总产量
    initial_sales_volume(crop_idx) = total_production * 0.8; % 计算该作物的初始销售量
end

% 显示重新计算完成的信息
disp('initial_sales_volume 重新计算完成。');

% 初始化未来年度的销售量、亩产量、种植成本和销售价格数组
sales_volume_data = zeros(num_crops, num_years);
yield_data = zeros(num_crops, num_years);
cost_data = zeros(num_crops, num_years);
price_data = zeros(num_crops, num_years);

% 为作物分类
grain_crops = {'小麦', '玉米'};         % 粮食作物
vegetable_crops = {'西红柿', '黄瓜'};     % 蔬菜作物
fungi_crops = {'羊肚菌', '香菇'};         % 菌类作物

% 定义销售量、产量、成本、价格增长或减少的函数
grain_sales_growth_rate = @(year) (1 + rand*0.05 + 0.05); % 粮食销售量增长
other_sales_change = @(year) (1 + rand*0.05 - 0.025);     % 其他作物销售量变化
yield_change = @(year) (1 + rand*0.1 - 0.05);             % 亩产量变化
cost_growth_rate = 1.05;                                  % 成本年增长率
vegetable_price_growth_rate = 1.05;                       % 蔬菜价格年增长率
fungi_price_decline_rate = @(crop) (0.95 * strcmp(crop, "羊肚菌") + (0.99 - rand*0.04) * ~strcmp(crop, "羊肚菌")); % 菌类价格变化

% 计算未来年度的销售量、亩产量、种植成本和销售价格
for year_idx = 1:num_years
    for crop_idx = 1:num_crops
        crop_name = crop_data.('作物名称'){crop_idx}; % 当前作物名称
        
        % 根据作物类型计算销售量
        if ismember(crop_name, grain_crops)
            sales_volume_data(crop_idx, year_idx) = initial_sales_volume(crop_idx) * grain_sales_growth_rate(years(year_idx));
        else
            sales_volume_data(crop_idx, year_idx) = initial_sales_volume(crop_idx) * other_sales_change(years(year_idx));
        end
        
        % 计算亩产量
        yield_data(crop_idx, year_idx) = initial_yield(crop_idx) * yield_change(years(year_idx));
        
        % 计算种植成本
        cost_data(crop_idx, year_idx) = planting_costs(crop_idx) * cost_growth_rate^year_idx;
        
        % 计算销售价格
        if ismember(crop_name, vegetable_crops)
            price_data(crop_idx, year_idx) = sale_prices(crop_idx) * vegetable_price_growth_rate^year_idx;
        elseif ismember(crop_name, fungi_crops)
            price_data(crop_idx, year_idx) = sale_prices(crop_idx) * fungi_price_decline_rate(crop_name)^year_idx;
        else
            price_data(crop_idx, year_idx) = sale_prices(crop_idx);
        end
    end
end

% 显示未来年度数据更新完成的信息
disp('未来年度的销售量、亩产量、种植成本和销售价格已成功更新。');
% 获取地块的数量
num_plots = height(plot_info);  

% 初始化最优种植方案数组（第一季和第二季）
optimal_plan_first_season = zeros(num_plots, num_crops, num_years);
optimal_plan_second_season = zeros(num_plots, num_crops, num_years);
total_revenue_per_year = zeros(num_years, 1);  % 初始化每年总收益数组

% 遍历未来每一年
for year_idx = 1:num_years
    
    % 初始化每块地每年每种作物的收益数组
    revenue_data = zeros(num_plots, num_crops);
    
    % 遍历每块地
    for plot_idx = 1:num_plots
        plot_area = plot_areas(plot_idx); % 获取当前地块面积
        
        % 遍历每种作物
        for crop_idx = 1:num_crops
            
            % 获取当前作物的亩产量、预期销售量、价格和成本
            actual_yield = yield_data(crop_idx, year_idx);
            expected_sales = sales_volume_data(crop_idx, year_idx);
            price = price_data(crop_idx, year_idx);
            cost = cost_data(crop_idx, year_idx);
            
            % 计算总产量
            total_production = actual_yield * plot_area;
            
            % 计算收入（取实际产量与预期销售量的较小值，并减去种植成本）
            revenue = min(total_production, expected_sales) * price - cost * plot_area;
            
            % 将当前作物的收入存储
            revenue_data(plot_idx, crop_idx) = revenue;
        end
        
        % 计算第一季最优种植作物索引和最大收入
        [max_revenue_first, best_crop_idx_first] = max(revenue_data(plot_idx, :));
        optimal_plan_first_season(plot_idx, best_crop_idx_first, year_idx) = plot_area; % 记录最优种植方案（第一季）
        
        % 计算第二季最优种植作物索引和最大收入
        [max_revenue_second, best_crop_idx_second] = max(revenue_data(plot_idx, :));
        optimal_plan_second_season(plot_idx, best_crop_idx_second, year_idx) = plot_area; % 记录最优种植方案（第二季）
        
        % 更新每年总收益
        total_revenue_per_year(year_idx) = total_revenue_per_year(year_idx) + max_revenue_first + max_revenue_second;
    end
end

% 显示最优种植方案已计算完成和每年总收益
disp('最优种植方案已计算完成，每年的总收益如下：');
disp(total_revenue_per_year);

% 初始化每年的种植计划结构体
num_years = length(years);  
num_plots = height(plot_info);  
num_crops = height(crop_data);  
yearly_plans = struct();  

% 预期销量系数和地块面积下限设置
expected_sales_factor = 0.8;  
min_plot_area = 0.1;  

% 初始化上一轮种植的作物数组
last_crop_planted = cell(num_plots, num_years);  

% 遍历未来每一年
for year_idx = 1:num_years
    
    % 初始化每一年第一季和第二季的种植方案
    year_plan_first_season = zeros(num_plots, num_crops);  
    year_plan_second_season = zeros(num_plots, num_crops); 

    % 遍历每块地
    for plot_idx = 1:num_plots
        plot_name = plot_info.('种植地块'){plot_idx}; % 获取地块名称
        plot_area = plot_info.('地块面积')(plot_idx); % 获取地块面积

        % 获取适用于当前地块的作物
        applicable_crops = result_table(strcmp(result_table.('地块类型'), plot_info.('地块类型'){plot_idx}), :);

        % 如果不是第一年，排除前一年已种植的作物
        if year_idx > 1
            last_crop = last_crop_planted{plot_idx, year_idx - 1};
            applicable_crops = applicable_crops(~strcmp(applicable_crops.('作物名称'), last_crop), :);
        end

        % 计算第一季适用的作物
        season_1_crops = applicable_crops(strcmp(applicable_crops.('季节'), '第一季'), :);
        if ~isempty(season_1_crops)
            % 遍历第一季作物
            for crop_idx = 1:height(season_1_crops)
                % 找出最佳作物（此处假设有一个find_best_crop函数）
                [best_crop_idx, ~] = find_best_crop(season_1_crops(crop_idx, :), crop_data, plot_area, expected_sales_factor, min_plot_area);
                year_plan_first_season(plot_idx, best_crop_idx) = plot_area;
                
                % 更新上一轮种植的作物
                last_crop_planted{plot_idx, year_idx} = crop_data.('作物名称'){best_crop_idx};
            end
        end

        % 计算第二季适用的作物
        season_2_crops = applicable_crops(strcmp(applicable_crops.('季节'), '第二季'), :);
        if ~isempty(season_2_crops)
            % 遍历第二季作物
            for crop_idx = 1:height(season_2_crops)
                % 找出最佳作物（此处假设有一个find_best_crop函数）
                [best_crop_idx, ~] = find_best_crop(season_2_crops(crop_idx, :), crop_data, plot_area, expected_sales_factor, min_plot_area);
                year_plan_second_season(plot_idx, best_crop_idx) = plot_area;
                
                % 更新上一轮种植的作物
                last_crop_planted{plot_idx, year_idx} = crop_data.('作物名称'){best_crop_idx};
            end
        end
    end
    
    % 保存每一年的种植计划
    yearly_plans(year_idx).first_season = year_plan_first_season;
    yearly_plans(year_idx).second_season = year_plan_second_season;
end

% 显示每年的种植方案计算完成
disp('每年的种植方案计算完成。');
% 定义豆类作物列表
legume_crops = ["黄豆", "豇豆", "芸豆", "红豆", "黑豆", "绿豆", "爬豆", "刀豆"];  

% 设置预期销售量系数和地块面积下限
expected_sales_factor = 0.9;  
min_plot_area = 0.3;  

% 初始化上一轮种植的作物数组和豆类作物种植年份数组
last_crop_planted = cell(num_plots, num_years);  
last_legume_year = zeros(num_plots, 1);  

% 遍历每一年
for year_idx = 1:num_years
    
    % 初始化每一年第一季和第二季的种植方案
    year_plan_first_season = zeros(num_plots, num_crops);  
    year_plan_second_season = zeros(num_plots, num_crops); 

    % 遍历每块地
    for plot_idx = 1:num_plots
        plot_name = plot_names{plot_idx}; % 当前地块名称
        plot_area = plot_areas(plot_idx); % 当前地块面积

        % 获取适用于当前地块的作物信息
        applicable_crops = result_table(strcmp(result_table.('地块类型'), plot_info.('地块类型'){plot_idx}), :);

        % 如果不是第一年，排除前一年已种植的作物
        if year_idx > 1
            last_crop = last_crop_planted{plot_idx, year_idx - 1};
            applicable_crops = applicable_crops(~strcmp(applicable_crops.('作物名称'), last_crop), :);
        end

        % 如果三年内未种植豆类作物，则优先选择豆类作物
        if (year_idx - last_legume_year(plot_idx)) >= 3
            crop_names = string(applicable_crops.('作物名称'));  

            % 筛选出豆类作物
            legume_crops_applicable = applicable_crops(ismember(crop_names, legume_crops), :);

            % 如果有豆类作物适用，则仅考虑豆类作物
            if ~isempty(legume_crops_applicable)
                applicable_crops = legume_crops_applicable;
            end
        end

        % 计算第一季适用的作物
        season_1_crops = applicable_crops(strcmp(applicable_crops.('季节'), '第一季'), :);
        total_planted_area_first_season = 0;
        if ~isempty(season_1_crops)
            % 遍历每种作物
            for crop_idx = 1:height(season_1_crops)
                % 调用find_best_crop函数找到最佳作物及其收益（此处假设函数存在）
                [best_crop_idx, best_revenue] = find_best_crop(season_1_crops(crop_idx, :), crop_data, plot_area, expected_sales_factor, min_plot_area);
                
                % 如果有最佳作物，且地块未完全种植
                if all(best_crop_idx > 0) && all(total_planted_area_first_season < plot_area)
                    planting_area = min(plot_area - total_planted_area_first_season, plot_area); 
                    
                    % 更新种植方案
                    year_plan_first_season(plot_idx, best_crop_idx) = planting_area;
                    total_planted_area_first_season = total_planted_area_first_season + planting_area;
                    
                    % 记录上一轮种植的作物
                    last_crop_planted{plot_idx, year_idx} = crop_data.('作物名称'){best_crop_idx};
                    
                    % 如果种植的是豆类作物，更新豆类作物种植年份
                    if ismember(string(crop_data.('作物名称')(best_crop_idx)), legume_crops)
                        last_legume_year(plot_idx) = year_idx;
                    end
                end
            end
        end
        
        % 如果地块未种植任何作物，随机选取一种作物种植
        if total_planted_area_first_season == 0 && ~isempty(season_1_crops)
            random_crop_idx = randi(height(season_1_crops)); 
            year_plan_first_season(plot_idx, random_crop_idx) = plot_area; 
        end

        % 计算第二季适用的作物
        season_2_crops = applicable_crops(strcmp(applicable_crops.('季节'), '第二季'), :);
        total_planted_area_second_season = 0;
        if ~isempty(season_2_crops)
            % 遍历每种作物
            for crop_idx = 1:height(season_2_crops)
                % 调用find_best_crop函数找到最佳作物及其收益（此处假设函数存在）
                [best_crop_idx, best_revenue] = find_best_crop(season_2_crops(crop_idx, :), crop_data, plot_area, expected_sales_factor, min_plot_area);
                
                % 如果有最佳作物，且地块未完全种植
                if all(best_crop_idx > 0) && all(total_planted_area_second_season < plot_area)
                    planting_area = min(plot_area - total_planted_area_second_season, plot_area); 
                    
                    % 更新种植方案
                    year_plan_second_season(plot_idx, best_crop_idx) = planting_area;
                    total_planted_area_second_season = total_planted_area_second_season + planting_area;
                    
                    % 记录上一轮种植的作物
                    last_crop_planted{plot_idx, year_idx} = crop_data.('作物名称'){best_crop_idx};
                    
                    % 如果种植的是豆类作物，更新豆类作物种植年份
                    if ismember(crop_data.('作物名称'){best_crop_idx}, legume_crops)
                        last_legume_year(plot_idx) = year_idx;
                    end
                end
            end
        end
        
        % 如果地块未种植任何作物，随机选取一种作物种植
        if total_planted_area_second_season == 0 && ~isempty(season_2_crops)
            random_crop_idx = randi(height(season_2_crops)); 
            year_plan_second_season(plot_idx, random_crop_idx) = plot_area; 
        end
    end

    % 保存每一年的种植计划
    yearly_plans = cell(num_years, 1);
    yearly_plans{year_idx} = struct('year', years(year_idx), ...
                                'first_season', year_plan_first_season, ...
                                'second_season', year_plan_second_season);
end

% 将每一年的种植方案导出到Excel文件
for year_idx = 1:num_years
    year_plan = yearly_plans{year_idx};
    
    % 生成有效且独特的作物名称
    valid_crop_names_first = matlab.lang.makeValidName(crop_names); 
    valid_crop_names_first = matlab.lang.makeUniqueStrings(valid_crop_names_first); 
    valid_crop_names_second = valid_crop_names_first; 
    valid_columns_first = ~cellfun('isempty', valid_crop_names_first);  
    valid_columns_second = valid_columns_first;  

    
    % 转换第一季种植方案为表格
    first_season_table = array2table(year_plan_first_season(:, valid_columns_first), ...
        'VariableNames', valid_crop_names_first(valid_columns_first), 'RowNames', plot_names);
    
    % 转换第二季种植方案为表格
    second_season_table = array2table(year_plan_second_season(:, valid_columns_second), ...
        'VariableNames', valid_crop_names_second(valid_columns_second), 'RowNames', plot_names);
    
    % 生成文件名
    first_season_filename = sprintf('最优种植方案2_第%d季_%d年.xlsx', 1, years(year_idx));
    second_season_filename = sprintf('最优种植方案2_第%d季_%d年.xlsx', 2, years(year_idx));
    
    % 将表格写入Excel文件
    writetable(first_season_table, first_season_filename, 'WriteRowNames', true);
    writetable(second_season_table, second_season_filename, 'WriteRowNames', true);
    end
    
    % 输出提示信息
    disp('所有最优种植方案已导出到 Excel 文件中。');