function [ndMat] = dctn(ndMat)
% DCTN receives N-dimensional array and computes nD-DCT transform.
% Developed by Seyed Mostafa Kia (seyedmostafa.kia@unitn.it), March, 2014.
n = ndims(ndMat);
for i = 1 : n
    s = size(ndMat);
    ndMat = reshape(ndMat,s(1),[]);
    ndMat = dct(ndMat);
    ndMat = reshape(ndMat,s);
    ndMat = shiftdim(ndMat,1);
end