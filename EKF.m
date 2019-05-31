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
f_x_j = @(x)[1,dt,0,0,0,0;0,1,0,0,0,0;0,0,1,dt,0,0;0,0,0,1,0,0;0,0,0,0,1,dt;0,0,0,0,0,1];
h_z_j = @(x)[x(1)/(x(1)^2 + x(3)^2 + x(5)^2)^(1/2),0,x(3)/(x(1)^2 + x(3)^2 + x(5)^2)^(1/2),0,x(5)/(x(1)^2 + x(3)^2 + x(5)^2)^(1/2), 0;
             -x(3)/(x(1)^2*(x(3)^2/x(1)^2 + 1)),0,1/(x(1)*(x(3)^2/x(1)^2 + 1)),0,0,0;
             -(x(1)*x(5))/((x(5)^2/(x(1)^2 + x(3)^2) + 1)*(x(1)^2 + x(3)^2)^(3/2)),0,-(x(3)*x(5))/((x(5)^2/(x(1)^2 + x(3)^2) + 1)*(x(1)^2 + x(3)^2)^(3/2)),0,1/((x(5)^2/(x(1)^2 + x(3)^2) + 1)*(x(1)^2 + x(3)^2)^(1/2)),0];

for iter = 1:TIMES
    % init state
    s = [1;1;1;1;1;1];
    x = s + f_w*(sigma_w*randn(3,1));
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
        % prediction
        p = f_x(x);
        A = f_x_j(x);
        pV(:,k) = p;
        P = A*P*A'+Q;
        %update
        z1 = h_z(p);
        H = h_z_j(p);
        K = P*H'*inv(H*P*H'+R);
        x = p+K*(z-z1);
        P = P-K*H*P;

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
