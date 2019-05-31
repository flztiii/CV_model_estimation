clc;
clear;

% define parameters
TIMES = 100;
N = 100;
n = 6;
m = 3;
dt = 0.1;
sigma_w = 0.1;
sigma_v = 0.01;
RMSEs = zeros(TIMES,N,1);

f_w = [dt^2/2.0, 0, 0; dt, 0, 0; 0, dt^2/2.0, 0; 0, dt, 0; 0, 0, dt^2/2.0; 0, 0, dt];
Q = f_w*(sigma_w^2*eye(3))*f_w';
R = sigma_v^2*eye(m);

f_x = @(x)[x(1)+dt*x(2); x(2); x(3)+dt*x(4); x(4); x(5)+dt*x(6); x(6)];
h_z = @(x)[sqrt(x(1)^2+x(3)^2+x(5)^2); atan(x(3)/x(1)); atan(x(5)/sqrt(x(1)^2+x(3)^2))];

for iter = 1:TIMES
    % init state
    s = [1;1;1;1;1;1];
    %x = s + f_w*(sigma_w*randn(3,1));
    x = s;
    P = eye(n);

    % records
    xV = zeros(n,N); % estimation
    sV = zeros(n,N); % true value
    zV = zeros(m,N); % observation
    pV = zeros(n,N); % prediction

    for k = 1:N
        % get values
        s = f_x(s) + f_w*(sigma_w*randn(3,1));
        z = h_z(s) + sigma_v*randn(3,1);
        sV(:,k) = s;
        zV(:,k) = z;
        %prediction
        [X, w_m, w_c] = sigmaPoints(x, P, n, 0.3, 2.0, 0.1);
        Y = zeros(n,2*n+1);
        for i = 1:2*n+1
            Y(:,i) = f_x(X(:,i));
        end
        [p, P_pre] = ut(Y, w_m, w_c, Q);
        pV(:,k) = p;
        %update
        Z = zeros(m,2*n+1);
        for i = 1:2*n+1
            Z(:,i) = h_z(Y(:,i));
        end
        [mu_z, P_z] = ut(Z, w_m, w_c, R);
        y = z - mu_z;
        K = zeros(n, m);
        for i = 1:2*n+1
            K = K + w_c(:,i)*(Y(:,i) - p)*(Z(:,i) - mu_z)';
        end
        K = K*inv(P_z);
        x = p + K*y;
        P = P_pre - K*P_z*K';

        xV(:,k) = x;
        RMSEs(iter,k,1) = sqrt(sum((x-s).*(x-s)));
    end
    
    if iter == 1
        xP = statePos(xV);
        sP = statePos(sV);
        pP = statePos(pV);

        FontSize = 14;
        LineWidth = 1;

        figure();
        % 画出真实值
        plot3(sP(1,:), sP(2,:), sP(3,:),'g-');
        hold on;

        % 画出最优估计值
        plot3(xP(1,:), xP(2,:), xP(3,:),'b-','LineWidth',LineWidth);
        hold on;

        % 画出状态测量值
        plot3(pP(1,:), pP(2,:), pP(3,:),'k+');
        hold on;

        legend('真实状态', 'EKF最优估计估计值','预测值');
        hold off;
    end
end

figure();
number = 1:N;
RMSEs = calcMeanError(RMSEs, N, TIMES);
plot(number,RMSEs);
title('100次蒙特卡洛仿真的RMSE');
xlabel('时间');
ylabel('RMSE');

function [X, w_m, w_c] = sigmaPoints(x, P, n, alpha, beta, kappa)
    lambda = alpha^2*(n+kappa)-n;
    len = 2*n+1;
    sigma = chol((n+lambda)*P);
    
    X = zeros(n, len);
    w_m = zeros(1, len);
    w_c = zeros(1, len);
    for i = 1:len
        if i == 1
            X(:,i) = x;
            w_m(:,i) = lambda/(lambda+n);
            w_c(:,i) = lambda/(lambda+n) + 1 - alpha^2 + beta;
        elseif i < n+2
            X(:,i) = x+sigma(:,i-1);
            w_m(:,i) = 1/(2*(lambda+n));
            w_c(:,i) = 1/(2*(lambda+n));
        else
            X(:,i) = x-sigma(:,i-n-1);
            w_m(:,i) = 1/(2*(lambda+n));
            w_c(:,i) = 1/(2*(lambda+n));
        end
    end
end

function [mu, S] = ut(X, W_m, W_c, M)
    [h, w] = size(X);
    mu = zeros(h,1);
    S = M;
    for i = 1:w
        mu = mu + W_m(:,i)*X(:,i);
    end
    for i = 1:w
        S = S + W_c(:,i)*(X(:,i) - mu)*(X(:,i) - mu)';
    end
end

function pos = statePos(state)
    [m,n] = size(state);
    pos = zeros(m/2,n);
    for k = 1:n
        pos(:,k) = [state(1,k);state(3,k);state(5,k)];
    end
end

function mean_error = calcMeanError(state, length, times)
    mean_error = [];
    for i = 1:length
        error = 0.0;
        for j = 1:times
            error = error + state(j,i,1);
        end
        error = error/times;
        mean_error = [mean_error, error];
    end
end
