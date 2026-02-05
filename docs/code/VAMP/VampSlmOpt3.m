% Options class for VampSlmEst3.m and VampSlmEst3_fastA.m
%
% USAGE
% -----
% "opt = VampSlmOpt3" uses default values
%
% "opt = VampSlmOpt3('property1',value1,'property2',value2, ...)" 
%       uses default values, except for the specified property/value pairs
%
% 

classdef VampSlmOpt3

    properties
        % need to specify if (and only if) "A" is a fxn-handle 
        Ah = [];        % fxn handle for A', i.e., adjoint of A
        N = [];         % number of columns in A

        % linear-solver parameters (for VampSlmEst_fastA.m only)
        solver = 'cg';  % in 'cg', 'lsqr', 'gd'
        solver_tol = 1e-4; % stopping tolerance of solver 
        solver_iters = 100;  % max solver iterations 
        solver_silent = true; 

        % SVD parameters (for VampSlmEst.m only)
        U = [];         % matrix of eigenvectors of A*A', or fxn handle
                        %   [U,D]=eig(A*A'); d=diag(D);
        d = [];         % vector of eigenvalues of A*A'
        Uh = [];        % fxn handle for U', needed only if U is a fxn handle
        UhA = [];       % fxn handle for U'*A
        AhU = [];       % fxn handle for A'*U, needed only if UhA is a fxn hndl


        % stopping parameters
        nitMax = 50;    % maximum number of VAMP iterations
        tol = 1e-4;     % stopping tolerance
        fxnStop = [];   % handle to a stopping fxn of the form
                        %   fxnStop = @(it,err,...
                        %               r1old,r1,gam1,x1,eta1,...
                        %               r2old,r2,gam2,x2,eta2) ...

        % damping
        damp = 0.97;    % damping parameter in (0,1] on r1 and 1/sqrt(gam1)
        damp_auto = false; % use auto-damping?
        divDamp2 = 1;   % damping on denoiser divergence estimates in (0,1)
                        % may help to decrease below 1!
        divDamp1 = 1;   % damping on lin-stage divergence estimates in (0,1)

        % initialization
        r1init = [];    % initial value of vector r1 (estimate of x)
        gam1init = [];  % initial value of scalar gam1 (precision on r1)
        gam2init = [];  % initial value of scalar gam2 (precision on r2)
                        % used only when iters_admm>0
        iters_admm = 0; % # of initial iters to fix gam1 & gam2
                                  % at their initial values, thereby running
                                  % Peaceman-Rachford ADMM

        % reporting
        verbose = 0;    % verbose reporting?
        histIntvl = 1;  % downsamples the verbose reporting and saved history

        silent = false; % disable all warnings 
        fxnErr = [];    % handle to a fxn of x2 for error reporting, e.g.,
                        %   fxnErr = @(x2) 10*log10( ...
                        %                    sum(abs(x2-xTrue).^2,1) ...
                        %                    ./sum(abs(xTrue).^2,1) );
        fxnGen = [];    % handle to a general fxn of the form
                        %   fxnGen = @(r1old,r1,gam1,x1,eta1,...
                        %              r2old,r2,gam2,x2,eta2)

        % monte-carlo divergence approximation
        divChange = 1e-3; % amount to perturb input for denoiser's Monte-Carlo
                        % divergence estimate
        divNum2 = 1;    % number of perturbations for denoiser divergence
                        % ...increase for a better estimate
        divNum1 = 1;    % number of perturbations for linear-stage divergence
                        % ...increase for a better estimate

        % noise-precision learning
        learnNoisePrec = false; % learn the noise precision?
        learnNoisePrecAlg = 'EM'; % learning type in {'EM','Newton'}
        learnNoisePrecNit = 100; % iterations used for noise learning
        learnNoisePrecTol = 1e-2; % tolerance used for noise learning 
        NoisePrecMax = 1e9; % max allowed value of noise precision 
        NoisePrecInit = [];% noise precision initialization (can leave empty)

        % keeping things sane
        gamMin = 1e-11; % minimum allowed precision 
        gamMax = 1e11;  % maximum allowed precision 

        % debugging
        x0 = [];        % where to supply the true value of x 
    end

    methods
        
        % Constructor with default options
        function opt = VampSlmOpt3(varargin)
            if nargin == 0
                % No custom parameters values, thus create default object
                return
            elseif mod(nargin, 2) == 0
                % User is providing property/value pairs
                names = fieldnames(opt);    % Get names of class properties

                % Iterate through every property/value pair, assigning
                % user-specified values.  Note that the matching is NOT
                % case-sensitive
                for i = 1:2:nargin-1
                    if any(strcmpi(varargin{i}, names))
                        % User has specified a valid property
                        propName = names(strcmpi(varargin{i}, names));
                        opt.(propName{1}) = varargin{i+1};
                    else
                        % Unrecognized property name
                        error(['VampSlmOpt3: %s is an unrecognized ' ...
                               'option'],num2str(varargin{i}));
                    end
                end
                return
            else
                error(['The VampSlmOpt3 constructor requires arguments ' ...
                    'be provided in pairs, e.g., VampSlmOpt3(''verbose'',' ...
                    ' false, ''nitMax'', 50)'])
            end
        end % VampSlmOpt3

    end % methods

end
