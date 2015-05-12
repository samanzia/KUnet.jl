classdef conv < layer
    properties
        b                           %bias vector, separated from w
        numOutputFilters            %number of output feature maps
        pooling                     %boolean, true if pooling needs to be done
        poolDim                     %if pooling true then the pooling scale  
        inputDim                    %dimension of input (square input assumed)
        filterDim                   %dimension of input filters
        poolFilter                  %pooling filter of ones 
    end
    
    properties (Transient = true)
        z                           %store pooling result if pooling true
        db                          %gradient for bias
    end
    
    methods
        
        function y = forw(l, x)
        % forw transforms input x to output y by convolving and 
        % passing through the activation function of the layer
        % and also performs pooling if true
            numImages = size(x, 4);
            convDim = l.inputDim - l.filterDim + 1;
            l.x = x;
            
            if(isempty(l.y))
                 if isa(x, 'gpuArray')
                    l.y = gpuArray.zeros(convDim,convDim,l.numOutputFilters,numImages);
                 else
                    l.y = zeros(convDim,convDim,l.numOutputFilters,numImages);
                 end
            else
                l.y = 0 * l.y;
            end
 
            if l.pooling
                if(isempty(l.z))
                    if isa(x, 'gpuArray')
                        l.z = gpuArray.zeros(convDim/l.poolDim,convDim/l.poolDim,l.numOutputFilters,numImages);
                    else
                        l.z = zeros(convDim/l.poolDim,convDim/l.poolDim,l.numOutputFilters,numImages);
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
                        l.z(:, :,filterNum,imageNum) = (downsample((downsample(conv2(l.y(:, :,filterNum,imageNum),filter,'valid'),l.poolDim))',l.poolDim))';
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
            
            for i = 1:numImages
                for f = 1:size(l.w,3)
                    for c = 1:size(l.w,4)
                        
                        if l.pooling
                            tempDx = bsxfun(@rdivide,kron(squeeze(dy(:,:,c,i)),ones(l.poolDim)),(1/(l.poolDim^2)));
                            tempDx = bsxfun(@rdivide,l.fback(tempDx),size(l.w,3));
                        else
                            tempDx = bsxfun(@rdivide,l.fback(dy(:,:,c,i)),size(l.w,3));
                        end
                        
                        l.dw(:,:,f,c) = l.dw(:,:,f,c) + conv2(l.x(:,:,f,i),rot90(tempDx,2),'valid');
                        l.db(c) = l.db(c) + sum(tempDx(:));
                        
                        if nargout > 0
                            dx(:,:,f,i) = dx(:,:,f,i) +conv2(tempDx,d.w(:,:,f,c),'full');
                        end
                        
                    end
                end
            end
        end %back
        
        function l = conv(varargin)
            l = l@layer(varargin{:});
        end
        
    end % methods
end %classdef conv < layer