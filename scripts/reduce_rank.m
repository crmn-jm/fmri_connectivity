    function [F_new,varsort,Tvalmin]=reduce_rank(F,Nvar,idl)    
    % F   : variable/mapa a reducir
    % Nvar: número de nodos a mostrar
    % idl : indices predefinidos bajo el que realizar el estudio (no obligatorio)
    
    % F_new   : mapa reducido
    % varsort : nodos relevantes ordenados
    % Tvalmin : umbral de corte
    
    % FUNCION: mantiene todas las conexiones relevantes dejando la matriz
    % simétrica pero elimina las relaciones entre las variables identicas
    
    F(isnan(F))=-Inf;
    
    if nargin<3
        
        [Fval,idmx]=sort(F(:),'descend'); % Ordena valores en orden descendente
        Tvalmin=Fval(Nvar); % Valor umbral: valor en la posicion Nvar 
        
        % Conversion posicion vector a matriz de los valores ordenados (se
        % repiten por pares,matriz simétrica)
        [X,Y]=ind2sub(size(F),idmx);
        varsort=[X(1:Nvar) Y(1:Nvar)]; 
        
        % Eliminar parejas que sean diagonal, misma region
        var=unique([X(1:Nvar) Y(1:Nvar)],'rows'); 
        
        % Mantener regiones sin repetir, al ser simetrica, es indiferente
        % coger la primera columna o la segunda, ya que se repiten los
        % valores por pares
        idxvar1=unique(var(:,1)); 
        %idxvar2=unique(var(:,2));
        
        % Generación de la matriz reducida
        F_new=zeros(length(idxvar1),length(idxvar1));
        
        % Rellenar F_new con la relación de una region con todas las otras
        % areas que han salido relevantes
        for i=1:length(idxvar1)                 
           F_new(i,:)=F(idxvar1(i),idxvar1);       
        end
        
        % Solo mostrar las relaciones que sean superiores al umbral
        F_new(F_new<Tvalmin)=0; 
        
    else
        
        [~,idmx1]=sort(F(idl,:),'descend');
        [~,idmx2]=sort(F(:,idl),'descend');
        
        var=unique([idmx1(1:Nvar) idmx2(1:Nvar)'],'rows');
        idxvar1=unique(var(:,1));
        %idxvar2=unique(var(:,2));
        F_new=diag(zeros(length(idxvar1),1));
        
        for i=1:length(idxvar1)
                 
           F_new(i,:)=F(idxvar1(i),idxvar1);
                 
        end
        F_new(F_new<Tvalmin)=0;
    end
        
    
    end