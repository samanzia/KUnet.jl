classdef conv < layer
    properties
        b                           %bias vector, separated from w
        numOutputFilters            %number of output feature maps
        pooling                     %boolean, true if pooling needs to be done
        poolDim                     %if pooling true then the pooling scale  
        inputDim                    %dimensions of input
        filterDim                   %dimension of input filters
        poolFilter                  %pooling filter of ones 
        
        db1                         % moving average of bias gradients for momentum
        db2                         % sum of squared bias gradients for adagrad
    end
    
    properties (Transient = true)
        z                           %store pooling result if pooling true
        db                          %gradient for bias
    end
    
    methods
        
        function dy = fback(l, dy,c,i)
        % fback multiplies its input with the derivative of the
        % activation function fforw.
            dy = dy .* l.y(:,:,c,i) .* (1 - l.y(:,:,c,i));
        end
        
        function y = forw(l, x)
        % forw transforms input x to output y by convolving and 
        % passing through the activation function of the layer
        % and also performs pooling if true
            numImages = size(x, 4);
            convDim1 = l.inputDim(1) - l.filterDim + 1;
            convDim2 = l.inputDim(2) - l.filterDim + 1;
            l.x = x;
            
            if(isempty(l.y))
                 if isa(x, 'gpuArray')
                    l.y = gpuArray.zeros(convDim1,convDim2,l.numOutputFilters,numImages);
                 else
                    l.y = zeros(convDim1,convDim2,l.numOutputFilters,numImages);
                 end
            else
                l.y = 0 * l.y;
            end
 
            if l.pooling
                if(isempty(l.z))
                    if isa(x, 'gpuArray')
                        l.z = gpuArray.zeros(convDim1/l.poolDim,convDim1/l.poolDim,l.numOutputFilters,numImages);
                    else
                        l.z = zeros(convDim1/l.poolDim,convDim1/l.poolDim,l.numOutputFilters,numImages);
                    end
                else
                    l.z = 0 * l.z;
                end
            end
                
            if(isempty(l.poolFilter))
                if isa(x, 'gpuArray')
                    l.poolFilter = gpuArray.ones(l.poolDim,l.poolDim)*(1/(l.poolDim*l.poolDim));
                else
                    l.poolFilter = ones(l.poolDim,l.poolDim)*(1/(l.poolDim*l.poolDim));
                end
            end
                
            for imageNum = 1:numImages
                for filterNum = 1:size(l.w,4)
                    for channelNum = 1:size(l.w,3)
                        l.y(:,:,filterNum,imageNum) = l.y(:,:,filterNum,imageNum) + ...
                            conv2(l.x(:, :,channelNum,imageNum),rot90(l.w(:,:,channelNum,filterNum),2),'valid');
                    end
                    l.y(:, :,filterNum,imageNum) = bsxfun(@rdivide,l.y(:, :,filterNum,imageNum),size(l.w,3)) ...
                        + l.b(filterNum);
                    l.y(:, :,filterNum,imageNum) = l.fforw(l.y(:, :,filterNum,imageNum));
                    
                    if l.pooling
                        l.z(:, :,filterNum,imageNum) = (downsample((downsample(conv2(l.y(:, :,filterNum,imageNum),l.poolFilter,'valid'),l.poolDim))',l.poolDim))';
                    end
                end
            end
            
            if l.pooling
                y = l.z;
            else
                y = l.y;
            end
        end %forw
        
        function dx = back(l, dy)
            if size(size(dy),2) < 4
                if l.pooling
                    dy = reshape(dy,size(l.z));
                else
                    dy = reshape(dy,size(l.y));
                end   
            end
            
            l.dw = 0 * l.w;
            l.db = 0 * l.b;
            
            if nargout > 0
                dx = 0 * l.x;
            end
            
            for i = 1:size(dy,4)
                for f = 1:size(l.w,3)
                    for c = 1:size(l.w,4)
                        
                        if l.pooling
                            tempDx = bsxfun(@times,kron(squeeze(dy(:,:,c,i)),ones(l.poolDim)),(1/(l.poolDim^2)));
                            tempDx = bsxfun(@rdivide,l.fback(tempDx,c,i),size(l.w,3));
                        else
                            tempDx = bsxfun(@rdivide,l.fback(dy(:,:,c,i),c,i),size(l.w,3));
                        end
                        
                        l.dw(:,:,f,c) = l.dw(:,:,f,c) + conv2(l.x(:,:,f,i),rot90(tempDx,2),'valid');
                        l.db(c) = l.db(c) + sum(tempDx(:));
                        
                        if nargout > 0
                            dx(:,:,f,i) = dx(:,:,f,i) +conv2(tempDx,l.w(:,:,f,c),'full');
                        end
                        
                    end
                end
            end
        end %back
        
        function update(l)
            if l.L1
                l.dw(:) = l.dw(:) + l.L1 * sign(l.w(:));
            end
            if l.L2
                l.dw(:) = l.dw(:) + l.L2 * l.w(:);
            end
            if l.adagrad
                if ~isempty(l.dw2)
                    l.dw2(:) = l.dw .* l.dw + l.dw2;
                    l.db2(:) = l.db .* l.db + l.db2;
                else
                    l.dw2 = l.dw .* l.dw; 
                    l.db2 = l.db .* l.db;
                end
                l.dw(:) = l.dw ./ (1e-8 + sqrt(l.dw2));
                l.db(:) = l.db ./ (1e-8 + sqrt(l.db2));
            end
            if ~isempty(l.learningRate)
                l.dw(:) = l.learningRate * l.dw;
                l.db(:) = l.learningRate * l.db;
            end
            if l.momentum
                if ~isempty(l.dw1)
                    l.dw1(:) = l.dw + l.momentum * l.dw1;
                    l.db1(:) = l.db + l.momentum * l.db1;
                else
                    l.dw1 = l.dw;
                    l.db1 = l.db;
                end
                if l.nesterov
                    l.dw(:) = l.dw + l.momentum * l.dw1;
                    l.db(:) = l.db + l.momentum * l.db1;
                else
                    l.dw(:) = l.dw1;
                    l.db(:) = l.db1;
                end
            end

            l.w(:) = l.w - l.dw;
            l.b(:) = l.b - l.db;

            if l.maxnorm
                norms = sqrt(sum(l.w.^2, 2));
                if any(norms > l.maxnorm)
                    scale = min(l.maxnorm ./ norms, 1);
                    l.w(:) = bsxfun(@times, l.w, scale);
                end
                
                norms = sqrt(sum(l.b.^2, 2));
                if any(norms > l.maxnorm)
                    scale = min(l.maxnorm ./ norms, 1);
                    l.b(:) = bsxfun(@times, l.b, scale);
                end
                
                
            end
        end
        
        function l = conv(varargin)
            l = l@layer(varargin{:});
        end
        
    end % methods
end %classdef conv < layer