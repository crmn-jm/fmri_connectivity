function [XS,SampleS]= Feature_Extraction(Y,X,Sample,ncomp,mode)
%%%%%%%%%% Variables %%%%%%%%%%%
% Y     : etiquetas
% X     : datos de entrenamiento
% Sample: datos de test
% ncomp : numero de componentes a reducir
% mode  : tipo de extraccion de caracteristicas
% 
% XS     : datos de entrenamiento reducidos
% SampleS: datos de test reducidos

%%%%%%%%%%%%%% Normalizacion de los datos %%%%%%%%%%%%%%%
% Aplicado solo al conjunto de entrenamiento
[X0,mu,sigma]=zscore(X);
Sample0 = bsxfun(@minus, Sample, mu);
Sample0 = bsxfun(@rdivide, Sample0, sigma);


if mode == 1 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           TSNE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    XS = tsne(X0, 'NumDimensions', ncomp);
    SampleS = tsne(Sample0, 'NumDimensions', ncomp, 'InitialY', XS(1:size(Sample0,1),:));
    
elseif mode == 2 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           PCA
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [COEFF,SCORE,latent,tsquare] = pca(X0, 'Economy', true);
    % coeff: eigenvectors, latent: eigenvalues
    XS = SCORE(:,1:ncomp); % X0*COEFF(:,1:ncomp);
    SampleS = Sample0*COEFF(:,1:ncomp);

    % figure;
    figure;
    scatter(XS(:,1), XS(:,2), 50, 'filled');
    xlabel('PC 1'); ylabel('PC 2');
    title('Principal components projection');
    grid on;
    
    % Gráfico de varianza explicada
    figure;
    explainedVar = 100 * latent / sum(latent);
    cumulativeVar = cumsum(explainedVar); % Suma acumulada de la varianza
    
    % Crear el primer eje Y (barras de varianza explicada)
    yyaxis left
    bar(explainedVar, 'FaceColor', [0.2 0.6 0.8]); % Barras en azul
    ylabel('Explained variance (%)');
    
    % Crear el segundo eje Y (línea de varianza acumulada)
    yyaxis right
    plot(1:length(cumulativeVar), cumulativeVar, '-o', 'Color', 'r', 'LineWidth', 2, 'MarkerFaceColor', 'r'); % Línea roja con marcadores
    ylabel('Cumulative variance (%)');
    
    % Configuración de ejes y título
    xlabel('Principal components');
    title('Variance explained by each component');
    legend({'Individual variance', 'Cumulative variance'}, 'Location', 'Best');
    grid on;

    
elseif mode == 3
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           PLS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [XL,yL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X0,Y,ncomp);
    W= stats.W;
    SampleS=Sample0*W;

    figure;
    gscatter(XS(:,1), XS(:,2), Y, 'br', 'o^', 8, 'filled'); % Colores azul y rojo, símbolos distintos
    xlabel('PLS Component 1');
    ylabel('PLS Component 2');
    title('Projection in PLS component space');
    legend({'Monolingual', 'Bilingual'}, 'Location', 'Best');
    grid on;

    % Varianza explicada
    figure;
    explainedVarPLS = PCTVAR(2,:) * 100;  % Varianza individual
    
    % Crear el primer eje (barras)
    yyaxis left
    bar(explainedVarPLS, 'FaceColor', [0.2 0.6 0.8]); % Barras en azul
    ylabel('Explained variance (%)');
    
    % Crear el segundo eje (línea de varianza acumulada)
    yyaxis right
    cumulativeVarPLS = cumsum(explainedVarPLS); % Varianza acumulada
    plot(1:length(cumulativeVarPLS), cumulativeVarPLS, '-o', 'Color', 'r', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    ylabel('Cumulative variance (%)');
    
    % Configuración de los ejes y título
    xlabel('Number of components');
    title('Variance Explained by PLS');
    grid on;
    legend({'Individual variance', 'Cumulative variance'}, 'Location', 'Best');





else
    error('No modo indicado en Feature_Extraction.m');
end

if any(isnan(SampleS(:)))
    error('Extraccion incorrecta en Feature_Extraction.m');
end


end
