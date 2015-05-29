function [K] = kron3D(X,Y)
%kron3D 3-D implementation of kron 2D matlab version
[mx,nx,ox] = size(X);
[my,ny,oy] = size(Y);

[ix,iy] = meshgrid(1:mx,1:my);
[jx,jy] = meshgrid(1:nx,1:ny);
[kx,ky] = meshgrid(1:ox,1:oy);

K = X(ix,jx,kx).*Y(iy,jy,ky);
end

