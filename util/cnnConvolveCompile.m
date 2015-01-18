%% compile cnnConvolve
% single thread version
% mexcmd = 'mex cc=g++ cnnConvolve.cc';
% eval(mexcmd);

mexcmd = ['mex cc=g++ cnnConvolve.cc CXXFLAGS="\$CXXFLAGS  -fopenmp"', ...
          ' LDFLAGS="\$LDFLAGS -fopenmp" -lgomp'];
eval(mexcmd);
 
%% compile maxPooling -> refer to maxPooling readme
