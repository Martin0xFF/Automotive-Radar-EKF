% Constant velocity motion model
function [X, MotionJacobi] = MotionModel(state, delta)
    MotionJacobi = eye(size(state,1));
    MotionJacobi(1, 3) = delta;
    MotionJacobi(2, 4) = delta;
    
    X = MotionJacobi*state;
end