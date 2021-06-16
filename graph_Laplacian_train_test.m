function [ L ] = graph_Laplacian_train_test( dataset_i,feature, M )

[N,n]=size(feature);

if dataset_i~=16
    feature=reshape(feature,[N 1 n]);
    c=reshape(feature-permute(feature,[2 1 3]),[N^2 n]);
    W=exp(-sum(c*M.*c,2));
else
    W=zeros(N^2,1);
    for i=1:n
        c0=reshape(feature(:,i)-feature(:,i)',[N^2 1]);
        W=W+exp(-sum(c0.*c0,2));
    end
end

W=reshape(W, [N N]);
W(1:N+1:end) = 0;
% W=fully_connected_to_knn(W,k);
% W_n0=W~=0;
% imagesc(W);
D=diag(sum(W));
L=D-W;
% L=D^(-0.5)*L*D^(-0.5);
L=(L+L')/2;
% L=L/max(eig(L));
end

