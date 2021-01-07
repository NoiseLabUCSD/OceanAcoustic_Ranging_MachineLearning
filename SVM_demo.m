% % SVM in MATLAB
% % Emma Reeves. May 1, 2017 (Modified May 4)
% % libsvm package can be found on github: https://github.com/cjlin1/libsvm
clear; clc;
%close all;
PLOT = 'on'; %(set PLOT == 'off' to suppress plotting)
Type = 'Classify'; %(set 'Classify' OR 'Regress')
Example = false; % test class example?
datapath = './data/';
DataSet = 'DataSet01';

% Generate Data for Class example
% partially taken from : http://www.alivelearn.net/?p=912

if Example
    rng(3);
    R = 1; %inner radius
    r = 0.9; %outer radius
    N = 2000;
    X = (4.*rand(N,2) - 2);
    Y = -1*ones(size(X,1),1);
    %pos = find(((d(:,1)+1).^2 + d(:,2).^2)<1 | ((d(:,1)-2).^2 + d(:,2).^2)<1); %two circles
    Y((X(:,1).^2 + X(:,2).^2)<=R.^2) = 1; % create middle circle
    pos = find(abs((X(:,1).^2 + X(:,2).^2)>R.^2) & (abs(X(:,1).^2 + X(:,2).^2)<r.^2)); % remove the points between two circles
    Y(pos) = [];
    X(pos,:) = [];
    
    X_test = X;
    X_train = X;
    Y_train = Y.';
    Y_test = Y_train;

else
    % use SBCEx16 data
    X_train = textread([datapath DataSet '/SBCEx16_training_input.txt']);
    Y_train = textread([datapath DataSet '/training_labels.txt']);
    X_test = textread([datapath DataSet '/SBCEx16_test_input.txt']);
    Y_test = textread([datapath DataSet '/test_labels.txt']);
    range_labels = textread([datapath DataSet '/Mapping_range_labels.txt']);
    
    N = size(X_test,1);
 end

% % % % EDIT: options % % % 
kernel_opts = {'-t 0','-t 2','-t 1','-t 3'};
titles = {'Linear Kernel','Radial Basis Function Kernel','Polynomial Kernel','Sigmoid Function Kernel'};
gamma = '0.0067'; % Gaussian parameter, use 1.8661 for example
c = '1'; % cost parameter, use 7.46 for example
kernel = kernel_opts{1};
    
    % svmtrain(Y, X, options). 
    % OPTIONS ==
        % The user should find the best parameters for their problem
        % (loop over values for -c, -g)
        % -c #: cost function 
        % -g #: gamma 
        % -q: quiet output to Command Window
        % -t #: 0 == linear, 1 == Poly, 2 == RBF, 3 == Sigmoid, 4 == custom
        
    switch Type
        case 'Classify'
            % train
                model = svmtrain(Y_train, X_train,['-c ' c ' -g ' gamma ' -q ' kernel]);
            % predict
                [y_pred,~, ~] = svmpredict(rand([N,1]), X_test, model,'-q'); %use dummy label inputs

        case 'Regress'
            % train
                model = svmtrain(Y_train, X_train,...
                    ['-s 4 -n 0.5 -h 0 -c ' c '-g ' gamma ' -q ',kernel]);
            % predict
                [y_pred,~, ~] = svmpredict(rand([N,1]), X_test, model,'-q'); % use dummy label inputs
    end 
    
%% Plot results (only plots in 2D)
if strcmp('on',PLOT)
    close(figure(1));
    figure(1); hold on
    if Example
        x_map = linspace(min(X_test(:))-0.25,max(X_test(:))+0.25,floor(length(X_test)/10));
        y_map = linspace(min(X_test(:))-0.25,max(X_test(:))+0.25,floor(length(X_test)/10));
        [X_map,Y_map] = meshgrid(x_map,y_map); % create a mesh
        boundary_map = [X_map(:), Y_map(:)];
        % plot the decision boundary
        boundary = svmpredict(rand([length(boundary_map),1]),boundary_map,model,'-q');
        pcolor(X_map,Y_map,reshape(boundary,[length(X_map),length(Y_map)]));
        shading flat;
        hold on;
        scatter(X_test(Y_test==jj-1,1),X_test(Y_test==jj-1,2),10,'ro','filled'); %plot the data
        scatter(X_test(Y==-1, 1),X_test(Y_test==-1, 2),10,'bo','filled');
        colormap([0.8 0.8 1; 1 0.8 0.8]);

        sv = full(model.SVs); % get and plot support vectors
        plot(sv(:,1),sv(:,2),'ko','linewidth',2,'MarkerSize',6);

        xlim([min(X_test(:))-0.25 max(X_test(:))+0.25]);
        ylim(xlim);
        xlabel('x1'); ylabel('x2');
        title(titles{strmatch(kernel,kernel_opts)});
    
    else
        plot(range_labels(Y_test),'r','linewidth',2);
        hold on;
        plot(range_labels(y_pred), 'bo');
        legend('GPS range','SVM Estimate');
        xlabel('Sample [index]');
        ylabel('Range (m)');
        title(['SVM Estimate of Ship Range using ' titles{strmatch(kernel,kernel_opts)}]);
        
    end
              
end
    
    