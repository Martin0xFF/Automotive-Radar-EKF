
% The State for this study is [ x, y, x_dot, x_dot ]'
% The measurement model for this study is [ x, y, vrad ]'
% We will assume all values are provided within the radar frame

function [Y, MeasureJacobi] = MeasurementModel(state, measure_dim)
    
% Extract state so it is human friendly
    x = state(1);
    y = state(2);
    xd = state(3);
    yd = state(4);
    
    % Estimate measurement using current state
    Y = [x; y; (x*xd + y*yd)/sqrt(x*x + y*y)];
    
    % Calculate Jacobian
    MeasureJacobi = zeros(measure_dim, size(state, 1));
    
    % First two rows are easy
    MeasureJacobi(1, 1) = 1;
    MeasureJacobi(2, 2) = 1;
    
    % Last Row is vrad and complicated
    MeasureJacobi(3, 1) = xd/sqrt(x*x + y*y) - (x*(x*xd + y*yd))...
        /((x*x + y*y)^(3/2));
    MeasureJacobi(3, 2) = yd/sqrt(x*x + y*y) - (y*(x*xd + y*yd))...
        /((x*x + y*y)^(3/2));
    MeasureJacobi(3, 3) = x/sqrt(x*x + y*y); 
    MeasureJacobi(3, 4) = y/sqrt(x*x + y*y); 
    
end