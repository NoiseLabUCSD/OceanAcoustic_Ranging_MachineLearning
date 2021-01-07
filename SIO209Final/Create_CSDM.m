
%%
%run two times to get training data and test data separately. 
% set flag TRAIN = true or false
% % % Edit this section % % % % % % % % % % %
DataSet = '01';
data_Dir = ['./data/DataSet' DataSet '/'];
TRAIN = false; %% if this is for training
filename = 'AcousticData_256093210_256094810';
FF=[305  319 333 347 361 372 389 403 417 431 447 459 471 487 497 515 ...
    529 543 557 571 585 596 613 628 641 655 669 683 697 711 ]; %select frequencies
sig_duration = 1;   % seconds
snapshot = 5; %number of sig_duration periods to average (1 - 20)
% % % % % % % % % % % % % % % % % % % % % % %


% read in data
x = textread([data_Dir filename]);

% default processing parameters
% fs = 25000; % sampling frequency
fs = 5000; % downsampling frequency
max_snap = floor(60/sig_duration);
deltf = fs/(sig_duration*fs);
ich = [17 19 20 21 22 23 24 25 26 27 28 29 30 31 32]; % selected hydrophones
nch = length(ich); % number of hydrophones
l = floor(sig_duration*fs); % length of sample, in samples
L = l*snapshot; % length of each CSDM, in samples
N = floor(size(x,1)/l) - snapshot; % number of CSDMs that fit in data

% preallocate variables for speed
i = 0;
xsnap = zeros(l, snapshot, nch);
CSDM = zeros(N, length(FF), nch, nch);

% process
while (L+(i+1)*l) <= length(x)
    i = i + 1;
    xsnap = reshape(x((1:L) + (i-1)*l,:),[l, snapshot, nch]); % reshape the data
    xsnap = squeeze(sum(xsnap,2)/snapshot); % average snapshots
    xsnap = fft(xsnap); % Fourier Transform
    for ifreq=1:length(FF)
        FREQ = FF(ifreq);
        Ind_f = floor(FREQ/deltf)+1;
        norm_data = xsnap(Ind_f,:) ./ repmat(sqrt(sum(abs(xsnap(Ind_f,:)).^2,2)),1,nch); %normalize @ each frequency
        CSDM(i,ifreq,:,:) = ( norm_data ).' * conj(norm_data); % create the cross-spectral density matrix (@ each frequency)
    end   
end

if ~TRAIN   %% for test data, we pick every 5 samples
    CSDM = CSDM(1:5:end,:,:,:);
end

%% Reshape the data for NN input (use only upper triangular)
input_data = [];
for ifreq=1:length(FF)
    for ii = 1:size(CSDM,3)
        input_data = [input_data squeeze(CSDM(:,ifreq,ii,ii:end))];
    end
end

% normalize to max and vectorize
norm = max(max(abs(input_data)));
p1 = input_data / norm;
x_output = [real(p1) imag(p1)];

if TRAIN
    save([data_Dir 'SBCEx16_training_input.txt'],'-ascii', 'x_output');
else
    save([data_Dir 'SBCEx16_test_input.txt'],'-ascii', 'x_output');
end

