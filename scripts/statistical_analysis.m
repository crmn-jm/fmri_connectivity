%%%%%%%%%%% statistical power analysis

% 2D Classification - False Positive Analysis
NUMK = 1000; % Number of iterations for cross-validation
Rvec_mono = matvec2D_red(1:Numc, :);
Rvec_bi = matvec2D_red(Numc+1:end, :);

% False Positives for Monolinguals
maphr = zeros([size(matvec2D_red, 2) size(matvec2D_red, 3)]);
mappr = zeros([size(matvec2D_red, 2) size(matvec2D_red, 3)]);

for k = 1:NUMK
    c = cvpartition(Numc, 'KFold', 2); % Split dataset into two
    disp(['Fold ' int2str(k)])
    for j = 1:size(Rvec_mono, 2)
        [h, p, ~, ~] = ttest2(Rvec_mono(c.test(1), j), Rvec_mono(~c.test(1), j), 'Alpha', 0.05);
        [idx1, idx2] = ind2sub([size(matvec2D_red, 2) size(matvec2D_red, 3)], j);
        maphr(idx1, idx2) = maphr(idx1, idx2) + h;
        mappr(idx1, idx2) = mappr(idx1, idx2) + p;
    end
end

%% FDR for Bilinguals
FDR = 0.05;

maphr = zeros([size(matvec2D_red, 2) size(matvec2D_red, 3)]);
mappr = zeros([size(matvec2D_red, 2) size(matvec2D_red, 3)]);

for k = 1:NUMK
    c = cvpartition(Numc, 'KFold', 2);
    disp(['Fold ' int2str(k)])
    for j = 1:size(Rvec_bi, 2)
        [h, p, ~, ~] = ttest2(Rvec_bi(c.test(1), j), Rvec_bi(~c.test(1), j), 'Alpha', 0.05);
        p_values(j) = p;
        q(j) = (j / size(Rvec_bi, 2)) * FDR;
    end
    
    % Sort p-values and compare with q for FDR
    [p_sort, index] = sort(p_values); % Sort p-values from smallest to largest
    comp = p_sort < q;
    comp_sort = comp(index);
    for l = 1:size(Rvec_bi, 2)
        [idx1, idx2] = ind2sub([size(matvec2D_red, 2) size(matvec2D_red, 3)], l);
        p_new = p_values(l);
        h_new

    end
end

