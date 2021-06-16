function [ class_train_test ] = class_vec_to_num( x,n_sample_full )
class_train_test=zeros(n_sample_full,1);
for i=1:n_sample_full
    vec_to_num=find(x(i,:)==max(x(i,:)));
    if length(vec_to_num)==1
       class_train_test(i)=vec_to_num;   
    end
    if length(vec_to_num)>1
       class_train_test(i)=vec_to_num(1);
    end
end
end

