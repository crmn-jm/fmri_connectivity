% Connectivity Study - Classificaton and Statistical Analysis

%% Loading Connectivity Matrices
Numr = 116;  % Number of regions
Nump = 220;  % Number of time-series
Numc = 29;   % Number of subjects per class

pathdata = 'path/to/data'; % Path to data folder
path_atlas = 'path/to/atlas'; % Path to atlas data

%% Obtaining Mean Connectivity Matrices
R1 = squeeze(mean(squeeze(R(1:Numc,:,:)), 1));  % Group 1
R2 = squeeze(mean(squeeze(R(Numc+1:end,:,:)), 1));  % Group 2

% Image dimensions
[rows1, cols1] = size(R1);
[rows2, cols2] = size(R2);

%% Conversion to Vector (Computationally Expensive)
Rvec = R(:,:);  % Convert 2D to 1D

%% Normality Check with Summary Statistics
normal_count1 = 0;  % Counter for Group 1
normal_count2 = 0;  % Counter for Group 2
total_tests = 0;    % Counter for total tests

% Randomly select nodes for visualization
num_nodes_to_plot = 9; % Number of nodes to plot
selected_nodes = randperm(size(Rvec, 2), num_nodes_to_plot);

for j = 1:size(Rvec, 2)
    % Extract data for each group
    group1 = Rvec(1:Numc, j);
    group2 = Rvec(Numc+1:end, j);
    
    % Exclude diagonal
    [idx1, idx2] = ind2sub([Numr, Numr], j);
    if idx1 == idx2
        continue
    end
    
    % Kolmogorov-Smirnov normality test
    [h1, ~] = kstest((group1 - mean(group1)) / std(group1));
    [h2, ~] = kstest((group2 - mean(group2)) / std(group2));
    
    % Count normal distributions
    if h1 == 0
        normal_count1 = normal_count1 + 1;
    end
    if h2 == 0
        normal_count2 = normal_count2 + 1;
    end
    total_tests = total_tests + 1;
end

% Calculate normality percentages
normal_percentage1 = (normal_count1 / total_tests) * 100;
normal_percentage2 = (normal_count2 / total_tests) * 100;

fprintf('Percentage of normal connections (Group 1): %.2f%%\n', normal_percentage1);
fprintf('Percentage of normal connections (Group 2): %.2f%%\n', normal_percentage2);

%% Parametric Study (t-test)
maph = zeros([Numr Numr]);
mapp = zeros([Numr Numr]);
mapT = zeros([Numr Numr]);
maph2 = zeros([Numr Numr]);
mapp2 = zeros([Numr Numr]);
mapT2 = zeros([Numr Numr]);

for j = 1:size(Rvec, 2)
    disp(['Computing t-test for connection ' int2str(j)])
    
    % Group comparison: monolingual < bilingual
    [h, p, ~, stats] = ttest2(Rvec(1:Numc, j), Rvec(Numc+1:end, j), 'Alpha', 0.01, 'tail', 'left');
    [idx1, idx2] = ind2sub([Numr, Numr], j);
    maph(idx1, idx2) = h;
    mapp(idx1, idx2) = p;
    mapT(idx1, idx2) = stats.tstat;
    
    % Group comparison: bilingual < monolingual
    [h, p, ~, stats] = ttest2(Rvec(Numc+1:end, j), Rvec(1:Numc, j), 'Alpha', 0.01, 'tail', 'left');
    maph2(idx1, idx2) = h;
    mapp2(idx1, idx2) = p;
    mapT2(idx1, idx2) = stats.tstat;
end

% Handle NaN values
mapT(isnan(mapT)) = 0;
mapT2(isnan(mapT2)) = 0;
maph(isnan(maph)) = 0;
maph2(isnan(maph2)) = 0;

% Final map (absolute t-statistics and significant results)
map = abs(mapT .* maph);
map2 = abs(mapT2 .* maph2);

%% Visualization (Circular Graph)
n_map = nnz(map);
n_map2 = nnz(map2);

labels = node_names2;
myColorMap = lines(length(labels));
Nvar = 50; % Number of connections to evaluate

% Select map to visualize: map (M < B) or map2 (M > B)
[mapTtest_red, var_ttest, Tvalmin_ttest] = reduce_rank(map, Nvar);
labels_red_ttest = labels(unique(var_ttest(:, 1)), :);  % Relevant regions
myColorMap_new = myColorMap(unique(var_ttest(:, 1)), :);
figure;
circularGraph(mapTtest_red, 'Colormap', myColorMap_new, 'Label', labels_red_ttest);


