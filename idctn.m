function [ndMat] = idctn(ndMat)
% IDCTN receives N-dimensional DCT coefficients and computes  inverse nD-DCT transform.
% Developed by Seyed Mostafa Kia (seyedmostafa.kia@unitn.it), March, 2014.
n = ndims(ndMat);
for i = 1 : n
    s = size(ndMat);
    ndMat = reshape(ndMat,s(1),[]);
    ndMat = idct(ndMat);
    ndMat = reshape(ndMat,s);
    ndMat = shiftdim(ndMat,1);
end