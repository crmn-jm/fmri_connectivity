function [cpVAF,cpVAFT,cpKVAF,cpKVAFT,cpKVAF2,cpKVAF2T,...
    cpRED,cpREDT,cpKRED,cpKREDT]=classifica(mat_vect,labels,modo)
% mat_vect : matriz de datos 
% labels   : etiquetas
% modo     : método de extraccion de caracteristicas, 1:TSNE 2:PCA 3:PLS
% 
% cpVAF    : rendimiento del clasificador con leave-one-out
% cpKVAF   : rendimiento del clasificador con k-fold
% cpKVAF   : rendimiento del clasificador normalizado con k-fold
% cpRED    : rendimiento del clasificador con leave-one-out y FES
% cpKRED   : rendimiento del clasificador con k-fold y FES

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clasificación basada en VAF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(14); % Semilla para selecciones aleatorias, ej. K-Fold

cpVAF = classperf(labels); %Inicialización de classperf
cpVAFT = classperf(labels);
N=numel(labels);
PC = 0; % 0:sin reduccion de datos 1:reduccion datos

for i=1:N
    % Leave-one-out
    Training_set=true(1,N);
    Training_set(i)=false;
    Testing_set=~Training_set;
    
    %Training
    Training=mat_vect(Training_set,:); % Si estuviera en 2D pasa a 1D
    Group=labels(Training_set);
    
    %Seleccion 2: de vóxeles mediante t-test (default) 
    %%% NOTA: no necesario cuando no se aplica en ordenador (baja memoria)
    if PC == 1
        Nselec=20;
        [IDX,~]=rankfeatures(Training',Group);
        Training=Training(:,IDX(1:Nselec));
    end

    SVMStruct = fitcsvm(Training,Group);
    
    %Testing
    if PC == 0
        Sample=mat_vect(Testing_set,:);
    else
        Sample=mat_vect(Testing_set,:);
        Sample=Sample(:,IDX(1:Nselec));
    end
    classout= predict(SVMStruct,Sample);
    disp(['Clase predicha: ' int2str(classout) ' Clase real: ' int2str(labels(i))]);
    
    classperf(cpVAF,classout,Testing_set);
    
    classout_t= predict(SVMStruct,Training);
    classperf(cpVAFT,classout_t,Training_set);
end

clear Training Group Sample Testing_set Training set SVMStruct classout
%% K-fold validation
K=10;
indices = crossvalind('Kfold',labels,K);
cpKVAF = classperf(labels);
cpKVAFT = classperf(labels);
for i = 1:K
    test = (indices == i); train = ~test;
    
    %Training
    if PC == 0
        Training=mat_vect(train,:);
        Group=labels(train);        
    else
        Training=mat_vect(train,:);
        Group=labels(train);
        Nselec=20;
        [IDX,~]=rankfeatures(Training',Group);
        Training=Training(:,IDX(1:Nselec));
        
    end
    
    SVMStruct = fitcsvm(Training,Group); % sin standarizar

    %Testing
    if PC == 0
        Sample=mat_vect(test,:);
    else
        Sample=mat_vect(test,:);
        Sample=Sample(:,IDX(1:Nselec));
    end
 
    classout = predict(SVMStruct,Sample);
    disp(['Ejecutándose Fold: ' int2str(i)]);  
    classperf(cpKVAF,classout,test);
    
    classout_t= predict(SVMStruct,Training);
    classperf(cpKVAFT,classout_t,train);
end

clear Training Group Sample SVMStruct classout

cpKVAF2 = classperf(labels);
cpKVAF2T = classperf(labels);
for i = 1:K
    test = (indices == i); train = ~test;
     
    %Training
    if PC == 0
        Training=mat_vect(train,:);
        Group=labels(train);        
    else
        Training=mat_vect(train,:);
        Group=labels(train);
        Nselec=20;
        [IDX,~]=rankfeatures(Training',Group);
        Training=Training(:,IDX(1:Nselec));
        
    end
      
    % SVM linear que normaliza las variables
    SVMStruct2 = fitcsvm(Training,Group,'Standardize',true);
    
    %Testing
    if PC == 0
        Sample=mat_vect(test,:);
    else
        Sample=mat_vect(test,:);
        Sample=Sample(:,IDX(1:Nselec));
    end
    
    classout = predict(SVMStruct2,Sample);
    disp(['Ejecutándose Fold: ' int2str(i)]);  
    classperf(cpKVAF2,classout,test);
    
    classout_t= predict(SVMStruct2,Training);
    classperf(cpKVAF2T,classout_t,train);
end

clear Training Group Sample SVMStruct classout

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clasificación Basada en EXTRACCION DE CARACTERISTICAS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Num_comp=3;
cpRED = classperf(labels);
cpREDT = classperf(labels);
symbols=['o' '*'];

% Leave-one-out
% for i=1:N
% 
%     Training_set=true(1,N);
%     Training_set(i)=false;
%     Testing_set=~Training_set;
% 
% 
%     Training=mat_vect(Training_set,:);
%     Group=labels(Training_set);
%     Sample=mat_vect(Testing_set,:);
% 
%     [Training,Sample]= Feature_Extraction(Group',Training,Sample,Num_comp,modo);
% 
%     % REPRESENTACION GRÁFICA DEL CONJUNTO MUESTRAL
% %     SCORE=[Training ; Sample]; %inicilaizo
% %     SCORE(Training_set,:)=Training;
% %     SCORE(Testing_set,:)=Sample;
% %     
% %     %Visualización de los scores
% %     figure
% %     scatter3(SCORE(labels==0,1),SCORE(labels==0,2),SCORE(labels==0,3));
% %     hold on
% %     scatter3(SCORE(labels==1,1),SCORE(labels==1,2),SCORE(labels==1,3),'r*');
% %     scatter3(SCORE(Testing_set,1),SCORE(Testing_set,2),SCORE(Testing_set,3),['g' symbols(labels(Testing_set)+1)]);
% %     legend('Class1','Class2','Prediction')
% 
% 
%     SVMStruct = fitcsvm(Training,Group); 
%     classout = predict(SVMStruct,Sample);
% 
%     disp(['Clase predicha: ' int2str(classout) ' Clase real: ' int2str(labels(i))]);
%     classperf(cpRED,classout,Testing_set);
% 
%     classout_t= predict(SVMStruct,Training);
%     classperf(cpREDT,classout_t,Training_set);
% end
% 
% clear Training Group Sample SVMStruct classout Testing_set Training_set

%% K-fold validation
K=10;
indices = crossvalind('Kfold',labels,K);
cpKRED = classperf(labels);
cpKREDT = classperf(labels);
for i = 1:K
    test = (indices == i); train = ~test;
    
    
    Training=mat_vect(train,:);
    Group=labels(train);
    Sample=mat_vect(test,:);
    
    if i < K
        close all;
    end
    [Training,Sample]= Feature_Extraction(Group',Training,Sample,Num_comp,modo);
    
    % REPRESENTACION GRÁFICA DEL CONJUNTO MUESTRAL
%     SCORE=[Training ; Sample]; %nos quedamos con los tres primeros coeficientes
%     SCORE(train,:)=Training;
%     SCORE(test,:)=Sample;
%     
%     %Visualización de los scores
%     figure
%     scatter3(SCORE(labels==0,1),SCORE(labels==0,2),SCORE(labels==0,3));
%     hold on
%     scatter3(SCORE(labels==1,1),SCORE(labels==1,2),SCORE(labels==1,3),'r*');
%     % Cambia de color las muestras de test
%     for s=1:sum(test)
%         idx=find(test);
%         scatter3(SCORE(idx(s),1),SCORE(idx(s),2),SCORE(idx(s),3),['g' symbols(labels(idx(s))+1)]);
%     end
%     legend('Class1','Class2','Prediction')
    
    %Training
    SVMStruct = fitcsvm(Training,Group);
    
    %Testing
    classout = predict(SVMStruct,Sample);
    disp(['Ejecutándose Fold: ' int2str(i)]);  
    classperf(cpKRED,classout,test);
    
    classout_t= predict(SVMStruct,Training);
    classperf(cpKREDT,classout_t,train);
end

end
