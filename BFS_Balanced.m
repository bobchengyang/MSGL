function [A_B,remove_mask]=BFS_Balanced(A) % input is the adjecency matrix of the given unbalanced graph removing self-loops. 
                             % After balancing the graph, one can add the self loops again (graph balancing does not affect to the self-loops) 

remove_mask=zeros(size(A));
G=graph(A);                             
v=bfsearch(G,1); % root node for BFS is 1.

IniClor=zeros(length(v),1); % Initial color assignment (no color, just 0 all elemnts). Here blue color is represeted as 1 and red color is represented as -1

 IniClor(1)=1; % assign blue color to node 1. 

for i=2:length(v)
    
   NeibNodes=find(A(v(i),:)); % neibouring nodes to node i
   CloredNodes=find(IniClor); % colored nodes in the current graph
   
   CloredNeibNodes=intersect(NeibNodes, CloredNodes); %neibouring colored nodes to node i
   
   DecesionValue=IniClor(CloredNeibNodes(1))*A(i,CloredNeibNodes(1));
   if DecesionValue>0
   IniClor(i)=1;                 % assign a color to node i.
   else
       IniClor(i)=-1;
   end
   
   for j=2:length(CloredNeibNodes)
       DecesionValue2=IniClor(CloredNeibNodes(j))*A(i,CloredNeibNodes(j))*IniClor(i); 
       if DecesionValue2<0
          A(i,CloredNeibNodes(j))=0;          % remove edges in the graph so that node i can be colored consistently according to CHT.
          A(CloredNeibNodes(j),i)=0;
          remove_mask(i,CloredNeibNodes(j))=1;
          remove_mask(CloredNeibNodes(j),i)=1;
       end
   
   end
           
end
A_B=A;
end
