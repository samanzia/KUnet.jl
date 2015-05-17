function y = forward(net, x, batch)
    ncol = size(x,size(size(x),2));
    if nargin < 3 batch = ncol; end
    y = [];
    for i=1:batch:ncol
        j = min(i+batch-1, ncol);
        a = x(:,:,:,i:j);       %changed for 4D input
        for l=1:numel(net)
            a = net{l}.forw(a);
        end
        y = [y, a];
    end
end
