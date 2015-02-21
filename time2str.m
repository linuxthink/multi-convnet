function str = time2str( totalS )
%time2str print time to string in 1h23m45.678s
% Input
%   s  : second, in float, by toc
% Output
%   str: 1h23m45.678s

d = floor(totalS / 86400);
totalS = totalS - d * 86400;

h = floor(totalS / 3600);
totalS = totalS - h * 3600;

m = floor(totalS / 60);
totalS = totalS - m * 60;

s = round(totalS);

% if d ~= 0
%   str = sprintf('%dd%dh%dm%fs', d, h, m, s);
% elseif h ~= 0
%   str = sprintf('%dh%dm%fs', h, m, s);
% elseif m ~= 0
%   str = sprintf('%dm%fs', m, s);
% else
%   str = sprintf('%fs', s);
% end

if d ~= 0
  str = sprintf('%dd%dh%dm%ds', d, h, m, s);
elseif h ~= 0
  str = sprintf('%dh%dm%ds', h, m, s);
elseif m ~= 0
  str = sprintf('%dm%ds', m, s);
else
  str = sprintf('%ds', s);
end

end
