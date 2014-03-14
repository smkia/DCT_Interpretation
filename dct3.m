function result=dct3(input)
% This function receives a 3D matrix as input and computes the 3D-DCT
% coefficients
[row,column,page]=size(input); 
coefficients_of_3d_dct=zeros(row,column,page); 
% 2D-DCT on first 2 dimensions
for i=1:page 
    middle_2d_matrix=input(:,:,i); 
    dct_coefficients=dct2(middle_2d_matrix); 
    coefficients_of_3d_dct(:,:,i)=dct_coefficients; 
end 
% 1D-DCT on the result of 2D-DCT
for i=1:row 
    for j=1:column 
        middle_vector=coefficients_of_3d_dct(i,j,:); 
        coefficients_of_dct1=dct(middle_vector); 
        coefficients_of_3d_dct(i,j,:)=coefficients_of_dct1(1,:); 
    end 
end 
% d = coefficients_of_3d_dct(1,1,1);
% coefficients_of_3d_dct = coefficients_of_3d_dct./d;
% coefficients_of_3d_dct(1,1,1)=d;
result=coefficients_of_3d_dct; 
end