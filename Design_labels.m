clear; clc;

% % % Edit this section % % % % % % % % % % %
data_Dir = './data/'; %your path to the data
DataSet = '01'; %which data set will you use (1-5)?
snapshot = 5; %number of sig_duration periods to average (1 - 20)
label_Range_out = (0:20:2960).'; % how to discretize the range(adjustable)
label_for_SVM = 1:length(label_Range_out);
% % % % % % % % % % % % % % % % % % % % % % %

%% prepare labels
% find files
data_Dir = [data_Dir 'DataSet' DataSet '/']; % add DataSet folder to path
folder = dir([data_Dir 'AcousticData*']); %acoustic data file names contain time stamps

% Parameters for 1.Training and 2.Test
type = {'training','test'};
Track_no = [1,2]; % track numbers
DT = [1,5]; % time spacing

filenames = {nan,nan}; % preallocate, training and test files
for ii = 1:2
    filenames{ii} = folder(ii).name; 
end
if strmatch('02',DataSet) % this data set is backwards 
   filenames = fliplr(filenames);
end

% load GPS data
load([ data_Dir 'GPS_Range_Time_DataSet' DataSet '.mat']);

for ii = 1:2
    % find the times from the acoustic data file name
    file = dir([data_Dir 'SBCEx16_' type{ii} '*.txt']);
    NumofSamples = size(load([data_Dir file.name]),1);
    Time1 = [str2double(filenames{ii}(14:16)) str2double(filenames{ii}(17:18)) ...
        str2double(filenames{ii}(19:20)) str2double(filenames{ii}(21:22))];
    Time2 = [str2double(filenames{ii}(24:26)) str2double(filenames{ii}(27:28)) ...
        str2double(filenames{ii}(29:30)) str2double(filenames{ii}(31:32))];
    
    % preallocate
   % y_test = zeros(NumofSamples,length(label_Range_out));
    track_Range_out = zeros(NumofSamples,1);
    y_test = zeros(NumofSamples,1);

    %create time vector
    x_t = datenum(0,0,Time1(1),Time1(2),Time1(3),...
        Time1(4)+(snapshot/2)):datenum(0,0,0,0,0,DT(ii)):datenum(0,0,Time2(1),Time2(2),Time2(3),Time2(4));

    %create range vector
    Range_t = interp1(GPS_Range_Time{Track_no(ii)}.time,GPS_Range_Time{Track_no(ii)}.range,x_t,'linear','extrap');
    Range_t = Range_t*1000; %convert to meters?
    
    % find the bin index
    for jj = 1:NumofSamples
        [val,idx]=min(abs(label_Range_out-Range_t(jj)));
        %y_test(jj,idx)=1;
        y_test(jj) = label_for_SVM(idx);
        track_Range_out(jj) = Range_t(jj);
    end
    
    % save labels and ranges
    save([data_Dir type{ii} '_labels.txt'],'-ascii', 'y_test');
    %save([data_Dir type{ii} '_labels_SVM.txt'],'-ascii', 'y_SVM'); 
    save([data_Dir type{ii} '_Ranges.txt'],'-ascii', 'track_Range_out');
end

% save discrete range
save([data_Dir 'Mapping_range_labels.txt'],'-ascii', 'label_Range_out');

