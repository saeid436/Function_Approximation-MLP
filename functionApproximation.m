% NN_HW_03_Saeid_Moradi_TakhminTabe

clc;
clear all;
close all;
%% Creat Tabe

Xdata = -3:0.1:3;
[r,c] = size(Xdata)

func = 0.1*(Xdata.^7) - 7*(Xdata.^3) ;

noise = 0.1 * max(func) * (rand(size(func))- 0.5); %% Creat Noise
noisyFunction = func+noise;  %Add Noise To Y

%% Creat Samples
bias = ones(r,c);
Samples = [bias;Xdata];
Targets = noisyFunction;

[randomizedSamples,randomizedTargets] = Randomizer(Samples,Targets,c); % Rsndomize Sample And Target

%% MLP For Training

H = input('Number Of Hidden Layers Neroun = ');

Weightih = .5 * (rand(2,H)-0.5); % Initial Values For Weights Between Input And Hidden Layers
Weightho = .5 * (rand(H+1,1)-0.5); % Initial Values For Weights Between Hidden Layer And Output Layer

Epoch = input('Number Of Epoch = '); % Number of Epoch
Alpha = 0.005; % Learning Rate
OutPut2 = zeros(c,1);
 
for ep=1:Epoch
    for i = 1:3:c
        
        rSample = randomizedSamples(:,i);
        rTarget = randomizedTargets(i);
        
        Netz = Weightih' * rSample;
        z = 1./(1+exp(-Netz));   % First Activity Function Is Sigmoid 
        Zbias = [1;z]; % Input Of OutPut Layer
        Nety = Weightho' * Zbias;
        OutPut = Nety*1;  % OutPut Activity Function Is Identity
        
        Deltay = rTarget - OutPut;
        EZbias = Weightho * Deltay;
        DeltaZbias = EZbias .* Zbias .* (1-Zbias);
        Deltaz = DeltaZbias(2:end);
        
        dWeightih = Alpha * rSample * Deltaz';
        dWeightho = Alpha * Zbias * Deltay';
        
        Weightih = Weightih + dWeightih; % Update Weightih
        Weightho = Weightho + dWeightho; % Update Weightho
    end
end

trainedW1=Weightih;
trainedW2=Weightho;
%% Test

for  i=1:c
    
        Netz2 = trainedW1' * [1;Xdata(i)];
        z2 = 1./(1+exp(-Netz2));   
        Zbias = [1;z2];
        nety2 = trainedW2' * Zbias;
        OutPut2(i) = nety2;  
end
approximatedFunction=OutPut2;
Error = Targets-approximatedFunction'; % Compute Error
%% Show Result

hold on

plot(Xdata,func,'k');       % Main Function

plot(Xdata,noisyFunction,'.');  % Noisy Function

plot(Xdata,approximatedFunction,'m'); % Approximated Function

stem(Xdata,Error,'c');   % Error

xlabel('Xdata')
ylabel('func  noisyFunction  approximatedFunction  Error')

grid on

legend('Main Func','Noisy Func','Approximated Func','Error')