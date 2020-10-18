clear;clc;close all;

xHx=Inf;
l=2;
H=[l^2 l^3 l^4 l^5;l^3 l^4 l^5 l^6;l^4 l^5 l^6 l^7;l^5 l^6 l^7 l^8];
while xHx>0
    x=randn(1,4);
    xHx=x*H*x';
end
disp('xHx is not PSD');