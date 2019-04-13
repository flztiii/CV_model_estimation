clc;
clear;

% define parameters
TIMES = 100;
RMSEs = zeros(TIMES,1);
N = 50;
n = 6;
m = 3;
dt = 0.1;
sigma_w = 0.5;
sigma_v = 0.01;

f_w = [dt^2/2.0, 0, 0; dt, 0, 0; 0, dt^2/2.0, 0; 0, dt, 0; 0, 0, dt^2/2.0; 0, 0, dt];
Q = f_w*(sigma_w^2*eye(3))*f_w';
R = sigma_v^2*eye(m);

f_x = @(x)[x(1)+dt*x(2); x(2); x(3)+dt*x(4); x(4); x(5)+dt*x(6); x(6)];
h_z = @(x)[sqrt(x(1)^2+x(3)^2+x(5)^2); atan(x(3)/x(1)); atan(x(5)/sqrt(x(1)^2+x(3)^2))];

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
    RMSE = 0;

    for k = 1:N
        % get values
        s = f_x(s) + f_w*(sigma_w*randn(3,1));
        z = h_z(s) + sigma_v*randn(3,1);
        sV(:,k) = s;
        zV(:,k) = z;
        % prediction
        [p, A] = jaccsd(f_x, x);
        pV(:,k) = p;
        P = A*P*A'+Q;
        %update
        [z1,H] = jaccsd(h_z, p);
        K = P*H'*inv(H*P*H'+R);
        x = p+K*(z-z1);
        P = P-K*H*P;

        xV(:,k) = x;
        RMSE = RMSE + sqrt(sum((x-s).*(x-s)));
        
    end
    RMSE = RMSE/N;
    RMSEs(iter) = RMSE;
    
    if iter == 1
        xP = statePos(xV);
        sP = statePos(sV);
        pP = statePos(pV);

        FontSize = 14;
        LineWidth = 1;

        figure();
        % ������ʵֵ
        plot3(sP(1,:), sP(2,:), sP(3,:),'g-');
        hold on;

        % �������Ź���ֵ
        plot3(xP(1,:), xP(2,:), xP(3,:),'b-','LineWidth',LineWidth);
        hold on;

        % ����״̬����ֵ
        plot3(pP(1,:), pP(2,:), pP(3,:),'k+');
        hold on;

        legend('��ʵ״̬', 'EKF���Ź��ƹ���ֵ','Ԥ��ֵ');
        hold off;
    end
end

figure();
number = 1:TIMES;
plot(number,RMSEs);
title('�ſ����ط����RMSE');
xlabel('ʵ�����');
ylabel('RMSE');

figure();
boxplot(RMSEs);
title('RMSE����ͼ');

function [z, A] = jaccsd(fun, x)
    % JACCSD Jacobian through complex step differentiation
    % [z J] = jaccsd(f,x)
    % z = f(x)
    % J = f'(x)
    %
    z = fun(x);
    n = numel(x);
    m = numel(z);
    A = zeros(m,n);
    h = n*eps;
    for k = 1:n
        x1 = x;
        x1(k) = x1(k)+h*1i;
        A(:,k) = imag(fun(x1))/h;
    end
end

function pos = statePos(state)
    [m,n] = size(state);
    pos = zeros(m/2,n);
    for k = 1:n
        pos(:,k) = [state(1,k);state(3,k);state(5,k)];
    end
end
