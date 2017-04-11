function Opt = FIG_USVClassify(varargin)

%% PARSE ARGUMENTS
P = parsePairs(varargin);

Opt.Journal = 'plos';
Dirs = setgetDirs;
Opt.Dir = Dirs.USVClassify;
Opt.FileFormat = 'pdf';
Opt.Resolution = 400;
