function result=idct3(input)
% This function computes inverse 3D-DCT on 3D input matrix of DCT
% coefficients
[row,column,page]=size(input); 
coefficients_of_3d_idct=zeros(row,column,page); 
% Inverse 2D-DCT
for i=1:page 
    middle_2d_matrix=input(:,:,i); 
    idct_coefficients=idct2(middle_2d_matrix); 
    coefficients_of_3d_idct(:,:,i)=idct_coefficients; 
end 
% Inverse 1D-DCT
for i=1:row 
    for j=1:column 
        middle_vector = coefficients_of_3d_idct(i,j,:); 
        coefficients_of_idct1=idct(middle_vector); 
        coefficients_of_3d_idct(i,j,:)=coefficients_of_idct1(1,:); 
    end 
end 
result=coefficients_of_3d_idct; 
 
             