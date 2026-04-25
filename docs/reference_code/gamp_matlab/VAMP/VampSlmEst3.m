function [x2,estFin,estHist] = VampSlmEst3(denoiser,y,A,opt)
%
% VampSlmEst3:  VAMP for the Standard Linear Model 
%
% DESCRIPTION
% -----------
% This implements VAMP for the standard linear model, 
% i.e., estimating a vector x from measurements 
%    y = A*x + w ,
% where w is additive white Gaussian noise and A is a known linear operator.
%
% VAMP-SLM is an iterative algorithm where each iteration has two stages.
% The first stage performs MMSE (linear) estimation of x under the prior 
% p(x)=N(x;r1,1/gam1) and the likelihood p(y|x)=N(y;A*x,1/gamwHat), 
% where gamwHat is (an estimate of) the noise precision.  This stage 
% yields an estimate x1 with precision eta1.  The pair (x1,eta1) is then 
% converted into the pair (r2,gam2), where r2 is treated as an 
% AWGN-corrupted version of the true x with noise precision gam2.  
% The second stage of the iteration denoises r2 to produce the estimate 
% x2 with precision eta2.  Finally, the pair (x2,eta2) is converted to 
% the pair (r1,gam1), concluding the iteration.  
% These iterations are repeated until r1 converges, and the final 
% value of x2 reported as the estimate of x.
%
%
% USAGE
% -----
% [x2,estFin,estHist] = VampSlmEst(denoiser,y,A,[opt])
%
% denoiser: estimates x from r = x + noise.
% Several options:
%   1) handle to fxn of form [xhat,xvar] = denoiser(r,rvar), where r is
%      the noisy signal and rvar is the noise variance
%   2) handle to fxn of form xhat = denoiser(r,rvar)
%   3) EstimIn object from GAMPmatlab
%
% y: an Mx1 vector of measurements (or MxL matrix in the MMV case) 
%
% A: linear operator.  Several options:
%   1) explicit matrix of size MxN
%   2) fxn handle, in which case opt.N must specify the # of columns in A
%      and opt.Ah must be a fxn handle to the adjoint operator
%   3) LinTrans object from GAMPmatlab
%
% opt: options structure, left empty or initialized via opt=VampSlmOpt3.
%   See VampSlmOpt3.m for details...
%
%
% IMPORTANT NOTE ON FAST OPERATORS
% --------------------------------
% For the linear estimation stage, VampSlmEst3 uses the eigendecomposition
% "[U,D]=eig(A*A')" to avoid the need to invert a matrix or solve
% a linear system at each iteration.  If U and d=diag(D) are not supplied,
% these quantities are computed, which is very expensive at high dimensions!
% Furthermore, if A is given as a function handle, then it is first
% converted to explicit matrix form, which may be undesirable when A is a
% fast operator.  These problems can be circumvented by supplying U and
% d=diag(D) to VampSlmEst3 through the opt structure.  Options are:
% 1) supply opt.U as an explicit MxM matrix and opt.d as an Mx1 vector,
% 2) supply opt.U and opt.Uh as fxn handles, and opt.d as an Mx1 vector,
% 3) use VampSlmOpt_fastA.m instead, which uses CG/LSQR/GD to solve at
%    each iteration.


% Process output format
if (nargout > 3)
    error('too many output arguments')
end
saveHist = (nargout >= 3);

% Get options
if (nargin < 3) || isempty(opt)
    opt = VampSlmOpt3();  % default options
end
nitMax = opt.nitMax;  % maximum number of iterations
tol = opt.tol;  % stopping tolerance
gamMin = opt.gamMin;  % minimum allowed precision
gamMax = opt.gamMax;  % maximum allowed precision
damp = opt.damp;  % damping parameter
damp_auto = opt.damp_auto;  % auto damping?
learnNoisePrec = opt.learnNoisePrec; % learn noise precision?
learnNoisePrecAlg = opt.learnNoisePrecAlg; % learning algorithm
learnNoisePrecNit = opt.learnNoisePrecNit; % nit for noise learning
learnNoisePrecTol = opt.learnNoisePrecTol; % tol for noise learning
gamwHatMax = opt.NoisePrecMax; % max allowed noise precision
verbose = opt.verbose; % verbose output? 
silent = opt.silent; % disable all warnings?
iters_admm = opt.iters_admm; % how many iterations of ADMM-PR before VAMP
dampDiv2 = opt.divDamp2;  % damping parameter for denoiser divergences

% Process measurements y
if isa(y,'numeric')
    [M,L] = size(y);
    if (~silent) && all(y(:)==zeros(M*L,1))
      warning('y is zero valued!  May cause problems...')
    end
else
    error('2nd input (y) must be a vector (or matrix) of measurements')
end

% Process linear transform A, length N, eigenvectors U, eigenvalues d
time_eig = nan;
if isa(A,'numeric')

    % treat the case where A is an explicit matrix
    [Mtest,N] = size(A);

    if Mtest~=M
        error('Number of rows in 2nd & 3rd inputs (y and A) do not match');
    end

    % if needed, compute eigendecomposition of A*A'
    if isempty(opt.U)||isempty(opt.d)
        if verbose
            fprintf('Computing eigendecomposition of A*Ah...\n');
            pause(eps);
        end
        tstart = tic;
          AAh = A*A';
          [U,D] = eig(0.5*(AAh+AAh'));
          d = diag(D);
          clear AAh D;
        time_eig = toc(tstart);
    else
        U = opt.U;
        d = opt.d;
        if length(d)~=M, error('Need length(d)==M'); end
    end

    % create function handles
    fxnA = @(x) A*x;
    fxnAh = @(z) A'*z;
    clear A;
    if isa(U,'function_handle')
      fxnU = U;
      if (~isempty(opt.Uh))&&isa(opt.Uh,'function_handle')
        fxnUh = opt.Uh;
      else
        error('Since opt.U is a fxn handle, opt.Uh must also be one')
      end
    else
      fxnU = @(x) U*x;
      fxnUh = @(x) U'*x;
      clear U;
    end

elseif isa(A,'function_handle') || any(strcmp(superclasses(A),'LinTrans'))

    % treat the case where A is a function handle or a LinTrans object
    if isa(A,'function_handle')

        fxnA = A;
        if isa(opt.Ah,'function_handle')
            fxnAh = opt.Ah;
        else
            error('opt.Ah must be a fxn handle to a linear operator');
        end
        if ~isempty(opt.N)
            if floor(opt.N)==opt.N
                N = opt.N;
            else
                error('opt.N must be an integer');
            end
        else
            error('Since 3rd argument is a fxn handle, must specify opt.N');
        end

    else % A is a LinTrans object

        fxnA = @(x) A.mult(x);
        fxnAh = @(x) A.multTr(x);
        [~,N] = size(A);

    end

    % test fxnA for size
    try
        z = fxnA(zeros(N,L));
        if any(size(z)~=[M,L])
            error('output size of 3rd input (A) doesnt match 2nd input (y)');
        end
        clear z;
    catch
        error('3rd input (A) doesnt accept inputs of correct size')
    end

    % test fxnAh for size
    try
        x = fxnAh(zeros(M,L));
        if any(size(x)~=[N,L])
            error('output size of opt.Ah doesnt match 2nd input (y)');
        end
        clear x;
    catch
        error('opt.Ah doesnt accept inputs of correct size')
    end

    % if needed, compute eigendecomposition of A*A'
    if isempty(opt.U)||isempty(opt.d)
        if verbose,
            fprintf('Computing eigendecomposition of A*Ah...\n');
            pause(eps);
        end
        tstart = tic;
          try
            AAh = fxnA(fxnAh(speye(M)));
          catch
            if verbose
                fprintf('fxnA doesnt support matrix argument!  Will slow down eigendecomposition...\n')
            end
            AAh = zeros(M);
            I = eye(M);
            for m=1:M, AAh(:,m) = fxnA(fxnAh(I(:,m))); end
          end
          [U,D] = eig(0.5*(AAh+AAh'));
          d = diag(D);
          clear AAh D;
        time_eig = toc(tstart);
    else
        U = opt.U;
        d = opt.d;
        if length(d)~=M, error('Need length(d)==M'); end
    end

    % create function handles for U and U'
    if isa(U,'function_handle')
      fxnU = U;
      if (~isempty(opt.Uh))&&isa(opt.Uh,'function_handle')
        fxnUh = opt.Uh;
      else
        error('Since opt.U is a fxn handle, opt.Uh must also be one')
      end
    else
      fxnU = @(x) U*x;
      fxnUh = @(x) U'*x;
    end
    clear U;

else
    error('3rd input must be a matrix or fxn handle to a linear operator')

end

% Load or create U'*A and A'*U handles
if isa(opt.UhA,'function_handle')
  fxnUhA = opt.UhA;
else
  fxnUhA = @(x) fxnUh(fxnA(x));
end
if isa(opt.AhU,'function_handle')
  fxnAhU = opt.AhU;
else
  fxnAhU = @(z) fxnAh(fxnU(z));
end

% Compute some constants
d = max(d,eps); % avoid zero-valued eigenvalues
s = sqrt(d); % singular values of A
Uhy = fxnUh(y); % U'*y where [U,D]=eig(A*A')

% Process denoiser
if isa(denoiser,'function_handle')
    if nargin(denoiser)~=2, error('need nargin(denoiser)==2'); end
    % check if this function handle returns [xhat,xvar] or just [xhat]
    try % to produce two outputs
        [~,~] = denoiser(zeros(N,L),ones(1,L));
        fxnDenoise = @(rhat,rvar) denoiser(rhat,rvar); 
        fxnDenoise_novar = @(rhat, rvar) first_output(denoiser, rhat, rvar); 
    catch % else turn into an EstimIn object
        denoiser1 = FxnhandleEstimIn(denoiser,...
                                     'changeFactor',opt.divChange,...
                                     'avg',opt.divNum2,'divMin',1e-5); 
        fxnDenoise = @(rhat,rvar) denoiser1.estim(rhat,rvar); 
        fxnDenoise_novar = denoiser; % no monte-carlo divergence!
    end 
elseif any(strcmp(superclasses(denoiser),'EstimIn')) 
    % turns EstimIn object into a function handle
    fxnDenoise = @(rhat,rvar) denoiser.estim(rhat,rvar); 
    fxnDenoise_novar = fxnDenoise;
else
    error(['First input (denoiser) must be either an EstimIn object, ',...
           'a fxn handle that accepts [rhat,rvar] and produces [xhat,xvar], ',...
           'or a fxn handle that accepts [rhat,rvar] and produces only xhat.']);
end 

% Process error-reporting function
if isa(opt.fxnErr,'function_handle')
    fxnErr = opt.fxnErr;
else
    fxnErr = [];
end

% Process general reporting function
if isa(opt.fxnGen,'function_handle')
    fxnGen = opt.fxnGen;
else
    fxnGen = [];
end

% Process stop function
if isa(opt.fxnStop,'function_handle')
    fxnStop = opt.fxnStop;
else
    fxnStop = [];
end

% Initialize noise precision & learning
gamwHatMin = M*L/norm(y,'fro')^2;
if ~isempty(opt.NoisePrecInit)
    gamwHat = opt.NoisePrecInit;
    if gamwHat>gamwHatMax
        warning('opt.NoisePrecInit too large')
        gamwHat = gamwHatMax;
    elseif gamwHat<gamwHatMin
        warning('opt.NoisePrecInit too small')
        gamwHat = gamwHatMin;
    end
else
    gamwHat = gamwHatMin;
end

% Prepare for saving history
if saveHist
    histIntvl = opt.histIntvl;
    nitSave = floor(nitMax/histIntvl);
    estHist.r1 = nan(N,L,nitSave);
    estHist.gam1 = nan(L,nitSave);
    estHist.x1 = nan(N,L,nitSave);
    estHist.gamwHat = nan(1,nitSave);
    estHist.eta1 = nan(L,nitSave);
    estHist.r2 = nan(N,L,nitSave);
    estHist.gam2 = nan(L,nitSave);
    estHist.x2 = nan(N,L,nitSave);
    estHist.eta2 = nan(L,nitSave);
    estHist.err = nan(L,nitSave);
    estHist.gen = nan(L,nitSave);
end

% Initialize r variables
if ~isempty(opt.r1init)
    r1 = opt.r1init;
    if size(r1,2)==1, r1 = r1*ones(1,L); end
else
    r1 = zeros(N,L); % default initialization
end
r1old = inf*ones(N,L); % used by fxnGen and fxnStop
r2old = inf*ones(N,L); % used by fxnGen and fxnStop

% Initialize gam1 precisions
if ~isempty(opt.gam1init)
    gam1 = opt.gam1init;
else
    %gam1 = sum(d.^2)*L/norm(bsxfun(@times,s,Uhy),'fro').^2; 
    gam1 = sum(abs(y).^2,1)/sum(abs(A(randn(N,1))).^2,1);
end
if size(gam1,2)==1, gam1 = gam1*ones(1,L); end

% Initialization specific to ADMM
if iters_admm > 0
    % initialize gam2 precision
    if ~isempty(opt.gam2init) 
        gam2 = opt.gam2init;
        if size(gam2,2)==1, gam2 = gam2*ones(1,L); end
    else
        gam2 = gam1;
    end
    eta1 = gam1 + gam2; 
    eta2 = eta1; 
    alf1 = gam1./eta1; 
    alf2 = gam2./eta2; 
end

% Set damping
damp1 = damp; % use same damping for R1 and gam1
damp2 = 1; % use same damping for R2 and gam2
damp_min = 0.05;

% ---------------------
% Run VAMP
err = nan(L,nitMax);
gen = nan(L,nitMax);
time = nan(1,nitMax);
gamwHat_ = nan(1,nitMax);
nitNoise = nan(1,nitMax);
tstart = tic;
i = 0;
stop = false;
flag = 0; 
%denoiser_calls = nan(1, nitMax); 

% Main loop of algorithm
while ~stop

    %-----update counter
    i = i + 1;

    if i <= iters_admm
        update_precisions = false;
    else
        update_precisions = true;
    end
    % disp(update_precisions);

    %-----damp r1 and gam1
    if (i > 1) && update_precisions 
        if damp_auto
          dampA = 2*min(alf1, alf2); % auto damping
          %dampA = 2*min(alf1,1-alf1);  % auto damping
          %dampA = 2*min(alf2,1-alf2);  % auto damping
          dampA1 = max(damp_min,min(dampA,damp1));
        else
          dampA1 = damp1;
        end
        r1 = bsxfun(@times,dampA1,r1) + bsxfun(@times,1-dampA1,r1old);
        gam1 = 1./(dampA1./sqrt(gam1) + (1-dampA1)./sqrt(gam1old)).^2;
    end
       
    %-----linear stage 
    UyAr1 = Uhy-fxnUhA(r1);
    gam1overD = (1./d)*gam1;

    if learnNoisePrec
       %-----learn noise precision 
       gam1overD_UyAr1_Sq = (gam1overD.^2).*abs(UyAr1).^2;
       switch learnNoisePrecAlg
         case 'EM'
           for ii=1:learnNoisePrecNit
             gamwHat_old = gamwHat;
             % note that resNormSq = sum(abs(y-A*x1).^2,1);
             resNormSq = sum(gam1overD_UyAr1_Sq./((gamwHat+gam1overD).^2));
             gamwHat = M/mean(resNormSq + sum(1./(gamwHat+gam1overD),1) );
             gamwHat = min(gamwHat,gamwHatMax);
             gamwHat = max(gamwHat,gamwHatMin);
             if abs(gamwHat_old-gamwHat)/gamwHat < learnNoisePrecTol, break; end
           end
           nitNoise(i)=ii;
         case 'Newton'
           hessReg = 1e-15; % Hessian regularization
           gamwHatMin = 1e-1; % minimum allowed value of gamwHat
           for ii=1:learnNoisePrecNit
             grad = -M/gamwHat +mean(sum(...
                  gam1overD_UyAr1_Sq./((gamwHat+gam1overD).^2) ...
                  + 1./(gamwHat+gam1overD) ));
             hess = M/gamwHat^2 -mean(sum(...
                  2*gam1overD_UyAr1_Sq./((gamwHat+gam1overD).^3) ...
                  + 1./((gamwHat+gam1overD).^2) ));
             gamwHat_old = gamwHat;
             gamwHat = gamwHat - grad/(abs(hess)+hessReg);
             gamwHat = min(max(gamwHat,gamwHatMin),gamwHatMax);
             if abs(gamwHat_old-gamwHat)/gamwHat < learnNoisePrecTol, break; end
           end
           nitNoise(i)=ii;
         otherwise
           error('unknown type of learnNoisePrecAlg')
       end
       if (~silent)&&(gamwHat<gamMin),
           warning('gamwHat=%g too small, iter=%i',gamwHat,i);
       end
    end % learnNoisePrec
    gamwHat_(i) = gamwHat;

    %-----update x and alf1
    x1 = r1 + fxnAhU(bsxfun(@rdivide,UyAr1,bsxfun(@plus,d,(1/gamwHat)*gam1)));
    if i>1, alf1old = alf1; end
    if update_precisions
        alf1 = 1 - (gamwHat/N)*sum(1./(gamwHat+gam1overD),1);
        eta1 = gam1./alf1; % reported but not used
    end

    %-----update r2 and gam2
    if i>1, r2old = r2; gam2old = gam2; end
    r2 = bsxfun(@rdivide,(x1-bsxfun(@times,alf1,r1)),1-alf1);
    if update_precisions
        gam2 = (1-alf1)./alf1.*gam1;
        if (~silent)&&any(gam2<gamMin)
            warning('gam2 too small, iter=%i',i);
        end
        if (~silent)&&any(gam2>gamMax)
            warning('gam2 too large, iter=%i',i);
        end
        gam2 = min(max(gam2,gamMin),gamMax);
    end

    %-----damp r2 and gam2
    if (i > 1) && update_precisions 
        r2 = bsxfun(@times,damp2,r2) + bsxfun(@times,1-damp2,r2old);
        gam2 = 1./(damp2./sqrt(gam2) + (1-damp2)./sqrt(gam2old)).^2;
    end
    
    %-----second half of iteration
    if i>1, alf2old = alf2; end
    if update_precisions
        % compute denoiser output and variance 
        [x2,x2var] = fxnDenoise(r2, 1./gam2);
        alf2 = mean(x2var,1).*gam2; 

        if i>1 
            alf2 = (dampDiv2*sqrt(alf2) + (1-dampDiv2)*sqrt(alf2old)).^2;
        end
        eta2 = gam2./alf2; % reported but not used
    else
        % compute denoiser output but not variance (to save computation)
        x2 = fxnDenoise_novar(r2, 1./gam2);
    end

    %-----record progress
    if ~isempty(fxnErr)
        err(:,i) = fxnErr(x2).'; % evaluate error function
    end
    if ~isempty(fxnGen)
        gen(:,i) = fxnGen(r1old,r1,gam1,x1,eta1,...
                          r2old,r2,gam2,x2,eta2); % evaluate general function
    end
    
    %-----save history
    if saveHist && rem(i,histIntvl)==0
        iHist = i/histIntvl;
        estHist.r1(:,:,iHist) = r1;
        estHist.gam1(:,iHist) = gam1;
        estHist.x1(:,:,iHist) = x1;
        estHist.gamwHat(iHist) = gamwHat;
        estHist.eta1(:,iHist) = eta1;
        estHist.r2(:,:,iHist) = r2;
        estHist.gam2(:,iHist) = gam2;
        estHist.x2(:,:,iHist) = x2;
        estHist.eta2(:,iHist) = eta2.';
        estHist.err(:,iHist) = err(:,i).';
        estHist.gen(:,iHist) = gen(:,i).';
    end
    
    %-----second half of iteration (continued)
    r1old = r1;
    r1 = bsxfun(@rdivide,(x2-bsxfun(@times,alf2,r2)),1-alf2);
    gam1old = gam1;
    if update_precisions
        gam1 = (1-alf2)./alf2.*gam2;
        gam1 = min(max(gam1,gamMin),gamMax);
    end

    if (~silent)&&any(gam1<gamMin)
        warning('gam1 too small, iter=%i',i);
    end
    if (~silent)&&any(gam1>gamMax)
        warning('gam1 too large, iter=%i',i);
    end
    
    %-----report progress
    if verbose && rem(i,histIntvl)==0
        if isscalar(dampR1), dampR1 = dampR1*ones(1,L); end
        if ~isempty(fxnErr)
              fprintf(['i=%3i: ',...
                  'dampR1=%5.3f ',...
                  'gam1=%8.3g ',...
                  'alf1=%7.4f ',...
                  'gam2=%8.3f ',...
                  'alf2=%7.4f ',...
                  'eta2/eta1=%6.2f ',...
                  'err=%8.3g\n'],...
                  [i*ones(L,1),...
                  dampR1.',...
                  gam1.',...
                  alf1.', ...
                  gam2.',...
                  alf2.',...
                  (eta2./eta1).',...
                  err(:,i)].');
        end
    end
    
    %-----stopping rule
    if ~isempty(fxnStop)
        stop = fxnStop(i,err(:,i),r1old,r1,gam1,x1,eta1,...
            r2old,r2,gam2,x2,eta2);
        if stop 
            flag = 1; 
        end
    end
    if all(sqrt(sum(abs(r1-r1old).^2)./sum(abs(r1).^2)) < tol)||(i>=nitMax)
        stop = true;
    end
    
    %-----measure time until iteration i
    time(i) = toc(tstart);
end % while

% Export Outputs
estFin.r1old = r1old;
estFin.r1 = r1;
estFin.gam1 = gam1;
estFin.x1 = x1;
estFin.gamwHat = gamwHat_(1:i);
estFin.nitNoise = nitNoise(1:i);
estFin.eta1 = eta1;
estFin.r2 = r2;
estFin.gam2 = gam2;
estFin.x2 = x2; % this is the main output: the final estimate of x
estFin.eta2 = eta2;
estFin.nit = i;
estFin.err = err(:,1:i);
estFin.gen = gen(:,1:i);
estFin.time = time(1:i);
%estFin.denoiser_calls = denoiser_calls(1:i); 
estFin.fxnStopFlag = flag; 

% Trim history
if saveHist
    iTrim = 1:floor(i/histIntvl);
    estHist.r1 = squeeze(estHist.r1(:,:,iTrim));
    estHist.gam1 = estHist.gam1(:,iTrim);
    estHist.x1 = squeeze(estHist.x1(:,:,iTrim));
    estHist.gamwHat = estHist.gamwHat(:,iTrim);
    estHist.eta1 = estHist.eta1(:,iTrim);
    estHist.r2 = squeeze(estHist.r2(:,:,iTrim));
    estHist.gam2 = estHist.gam2(:,iTrim);
    estHist.x2 = squeeze(estHist.x2(:,:,iTrim));
    estHist.eta2 = estHist.eta2(:,iTrim);
    estHist.err = estHist.err(:,iTrim);
end

end % VampSlmWhEst



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = first_output(f, r, rvar)
    [x, ~] = f(r, rvar); 
end


