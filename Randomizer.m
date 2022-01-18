
function [RandomSample,RandomTarget] = Randomizer(Sample,Target,c)
    
    RandomSample = zeros(2,c);
    RandomTarget = zeros(1,c);
    
    RandIndex = randperm(c);
    
    for i = 1:c
        StarIndex = RandIndex(i);
        RandomSample(:,StarIndex) = Sample(:,i);
        RandomTarget(:,StarIndex) = Target(:,i);
    end
    
end

