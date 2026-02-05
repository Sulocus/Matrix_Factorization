function [xhat,iters,state] = LinSolve(y,A,AH,r,rho,init,opt)
%% LinSolve: solves regularized Linear System using an iterative algorithm
%  xhat = argmin_x {0.5*||A*x-y||^2 + 0.5*rho*||x-r||^2}
%
% USAGE
% -----
% [xhat,iters,state] = LinSolve(y,A,AH,r,rho[,init,opt])
% 
% xhat: approximate solution to regularized linear system
% iters: # iterations before algorithm reached stopping tolerance
% state: used to warm-start by setting init=state
%
% INPUTS
% ------
% y: an Mx1 vector of measurements (or MxL matrix in MMV case)
% A : fxn handle for the forward operator
% AH: fxn handle for the adjoint operator
% r: an Nx1 vector of noisy input  (or NxL matrix in MMV case)
% rho: scalar (or 1xL vector in MMV case)
% init: initialization (use init=state for warm-start)

% opt: optional arguments with fields
%      a) 'solver': in 'lsqr' (default), 'cg', 'gd'
%      b) 'solver_iters' : default =100
%      c) 'solver_tol'   : default =1e-5
%      d) 'solver_silent': default =true

% handle initialization
if (nargin<6) || isempty(init)
    init = 0;
end


% default options
solver = 'lsqr'; % default solver
solver_iters = 100; % default # of iterations
solver_tol = 1e-5; % default stopping tolerance
solver_silent = true;

% replace default options if needed
if (nargin>=7) && ~isempty(opt)
    solver       = getVal(opt, 'solver', solver);
    max_iters    = getVal(opt, 'solver_iters', solver_iters);
    tol          = getVal(opt, 'solver_tol', solver_tol);
    silent       = getVal(opt, 'solver_silent', solver_silent);
end

% determine dimensions
[N,L] = size(r); 
M = size(y, 1); 

% expand inputs if needed
if isscalar(rho)
    rho = rho * ones(1, L); 
end
if size(init,2) == 1
    init = init * ones(1, L);
end
if size(init,1) == 1
    if strcmp(solver,'gd') && M<N
        init = ones(M,1) * init; 
    else
        init = ones(N,1) * init;
    end
end

% precompute for relative residual
compute_relres = false;
if compute_relres
    if M>=N
        norm2 = sum(abs(y).^2,1) + rho.*sum(abs(r).^2,1);
    else
        Ar = A(r);
        norm2 = sum(abs(AH(y)).^2,1) + rho.*sum(abs(Ar).^2,1);
    end
end

% prepare for debugging
debug = false;
if debug
    if M>=N
        cost = @(x) sum(abs(y-A(x)).^2,1) + rho.*sum(abs(x-r).^2,1); 
    else
        Ar = A(r);
        cost = @(v) sum(abs(AH(y-v)).^2,1) + rho.*sum(abs(v-Ar).^2,1);
    end
end

%--------------------------------------------------
% Solve the linear system
switch solver
    case 'gd'
      if M >= N
        % solve xhat = argmin_x {0.5||A*x-y||^2 + 0.5*rho||x-r||^2}
        xhat = init; 
        Axhat = A(xhat);
        flag = 1; 
        for it = 1:max_iters
            
            grad = AH(Axhat-y) + bsxfun(@times,xhat-r,rho);
            mu_opt = 1./(rho + min(1e15,sum(abs(A(grad)).^2,1)./sum(abs(grad).^2,1)) ); 
            if debug, xhat_old = xhat; end
            xhat = xhat - bsxfun(@times,mu_opt,grad);
            Axhat = A(xhat);
            
            if debug % try line search
                K = 5;
                J = nan(K,L);
                scale = logspace(-0.1,0.1,K);
                for k=1:K
                    xhat_k = xhat_old - bsxfun(@times,scale(k)*mu_opt,grad);
                    J(k,:) = cost(xhat_k);
                end
                J_avg = mean(J,2) % report average cost over L columns
                % verify that middle entry is lowest!
            end
            
            % relative residual as defined in LSQR
            if compute_relres
                relres2 = (sum(abs(y-Axhat).^2,1)+rho.*sum(abs(r-xhat).^2,1))./norm2;
            end
            
            % stopping condition
            it_min = 0;
            err = sqrt(sum(abs(grad).^2,1));
            if (it>it_min) && all(err < tol) 
                flag = 0; 
                break
            end
        end %it
        iters = it; 
        state = xhat;
        if (~silent) && flag
            warning(['GD flag=','not converged']); 
        end
      else % if M < N
        % solve vhat = argmin_v {0.5||A'*(v-y)||^2 + 0.5*rho||v-A*r||^2}
        % then set xhat = r + A'*(y-vhat)/rho
        vhat = init;
        AHvy = AH(vhat-y);
        flag = 1;
        for it = 1:max_iters
            
            grad = bsxfun(@times,vhat,rho) + A(AHvy-bsxfun(@times,r,rho));
            mu_opt = 1./(rho + min(1e15,sum(abs(AH(grad)).^2,1)./sum(abs(grad).^2,1)) );
            if debug, vhat_old = vhat; end
            vhat = vhat - bsxfun(@times,mu_opt,grad);
            AHvy = AH(vhat-y);
            
            if debug % try line search
                K = 5;
                J = nan(K,L);
                scale = logspace(-0.1,0.1,K);
                for k=1:K
                    vhat_k = vhat_old - bsxfun(@times,scale(k)*mu_opt,grad);
                    J(k,:) = cost(vhat_k);
                end
                J_avg = mean(J,2) % report average cost over L columns
                % verify that middle entry is lowest!
            end
            
            % relative residual as defined in LSQR
            if compute_relres
                relres2 = (sum(abs(AHvy).^2,1)+rho.*sum(abs(Ar-vhat).^2,1))./norm2
            end
            
            % stopping condition
            it_min = 0;
            err = sqrt(sum(abs(grad).^2,1));
            if (it>it_min) && all(err < tol)
                flag = 0; 
                break
            end
            
        end %it
        xhat = r + bsxfun(@times,AH(y-vhat),1./rho);
        state = vhat;
        iters = it; 
        if (~silent) && flag
            warning(['GD flag=','not converged']); 
        end
        
      end % if M<N

    case 'lsqr'
        sqrtRho = sqrt(rho); 
        AA = @(x) [reshape(A(reshape(x,[N,L])),[],1);...
                   reshape(bsxfun(@times,reshape(x,[N,L]),sqrtRho),[],1)]; 
        AAH = @(z) reshape(AH(reshape(z(1:M*L),[M,L])),[],1) + ... 
                   reshape(bsxfun(@times,reshape(z(M*L+1:end),[N,L]),sqrtRho),[],1); 
        A_lsqr = @(x,mode) Amatlab(x,mode,AA,AAH); % see fxn at end of this file
        b_lsqr = [y(:); reshape(bsxfun(@times,r,sqrtRho),[],1)];
        
        [xhat,flag,~,iters] = lsqr(A_lsqr,b_lsqr,tol,max_iters,[],[],init(:)); % notice warm start
        xhat = reshape(xhat,[N,L]);
        state = xhat;
        if (~silent) && flag
            warning(['LSQR flag=',flag]); 
        end

    case 'cg'
        % note cg wants the forward operator to be symmetric
        A_cg = @(x) reshape(AH(A(reshape(x,[N,L]))),[],1) + ...
                    reshape(bsxfun(@times,reshape(x,[N,L]),rho),[],1); 
        b_cg = reshape(AH(y),[],1) + reshape(bsxfun(@times,r,rho),[],1); 

        [xhat,flag,~,iters] = pcg(A_cg,b_cg,tol,max_iters,[],[],init(:));
        xhat = reshape(xhat,[N,L]);
        state = xhat;
        if (~silent) && flag
            warning(['PCG flag=',flag]); 
        end

    case 'cg2'
        % note conjgrad2 wants the forward operator to be symmetric
        A_cg = @(x) reshape(AH(A(reshape(x,[N,L]))),[],1) + ...
                    reshape(bsxfun(@times,reshape(x,[N,L]),rho),[],1); 
        b_cg = reshape(AH(y),[],1) + reshape(bsxfun(@times,r,rho),[],1); 

        xhat = reshape( conjgrad2(A_cg,b_cg,init(:),tol), [N,L]);
        state = xhat;
        iters = size(b_cg,1); 

    otherwise
        error('unrecognized solver')
end
end % LinSolve

% Create Matlab wrapper function
function out = Amatlab(in,mode,A,Ah)
    if strcmp(mode,'transp')
        out = Ah(in);
    else
        out = A(in);
    end
end

function out = getVal(opt,field,default)
    out = default;
    if isprop(opt, field) || isfield(opt, field)
        out = opt.(field); 
    end
end
