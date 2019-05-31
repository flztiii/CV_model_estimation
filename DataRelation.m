clc;
clear;

% define parameters
TIMES = 100;  %蒙特卡洛仿真次数
RMSEs = zeros(TIMES,1);
N = 100;
n = 6;
m = 3;
dt = 0.1;
sigma_w = 0.1;  % 状态转移噪声
sigma_v = 0.01;  % 传感器噪声
OBJ = 3;  % 观测目标数量
SENSOR = 2;  % 传感器数量

f_w = [dt^2/2.0, 0, 0; dt, 0, 0; 0, dt^2/2.0, 0; 0, dt, 0; 0, 0, dt^2/2.0; 0, 0, dt];
Q = f_w*(sigma_w^2*eye(3))*f_w';
R = sigma_v^2*eye(m);

f_x = @(x)[x(1)+dt*x(2); x(2); x(3)+dt*x(4); x(4); x(5)+dt*x(6); x(6)];
h_z = @(x)[sqrt(x(1)^2+x(3)^2+x(5)^2); atan(x(3)/x(1)); atan(x(5)/sqrt(x(1)^2+x(3)^2))];
f_x_j = @(x)[1,dt,0,0,0,0;0,1,0,0,0,0;0,0,1,dt,0,0;0,0,0,1,0,0;0,0,0,0,1,dt;0,0,0,0,0,1];
h_z_j = @(x)[x(1)/(x(1)^2 + x(3)^2 + x(5)^2)^(1/2),0,x(3)/(x(1)^2 + x(3)^2 + x(5)^2)^(1/2),0,x(5)/(x(1)^2 + x(3)^2 + x(5)^2)^(1/2), 0;
             -x(3)/(x(1)^2*(x(3)^2/x(1)^2 + 1)),0,1/(x(1)*(x(3)^2/x(1)^2 + 1)),0,0,0;
             -(x(1)*x(5))/((x(5)^2/(x(1)^2 + x(3)^2) + 1)*(x(1)^2 + x(3)^2)^(3/2)),0,-(x(3)*x(5))/((x(5)^2/(x(1)^2 + x(3)^2) + 1)*(x(1)^2 + x(3)^2)^(3/2)),0,1/((x(5)^2/(x(1)^2 + x(3)^2) + 1)*(x(1)^2 + x(3)^2)^(1/2)),0];

% 第一个传感器的records
sensor1_xV = zeros(OBJ,TIMES,N,n); % estimation
sensor1_zV = zeros(OBJ,TIMES,N,m); % observation
sensor1_pV = zeros(OBJ,TIMES,N,n); % prediction
sensor1_RMSE = zeros(OBJ,TIMES,N,1); % rmse
% 第二个传感器的records
sensor2_xV = zeros(OBJ,TIMES,N,n); % estimation
sensor2_zV = zeros(OBJ,TIMES,N,m); % observation
sensor2_pV = zeros(OBJ,TIMES,N,n); % prediction
sensor2_RMSE = zeros(OBJ,TIMES,N,1); % rmse
% 融合结果的record
final_xV = zeros(OBJ,TIMES,N,n); % estimation
final_RMSE = zeros(OBJ,TIMES,N,1); % rmse
% 真实轨迹的record
true_sV = zeros(OBJ,TIMES,N,n); % true value

for iter = 1:TIMES
    % 第一个物体初始化
    s1 = [1;1;0;1.5;0;1.5];
    sensor1_x1 = s1 + f_w*(sigma_w*randn(3,1));
    sensor1_P1 = eye(n);
    sensor2_x1 = s1 + f_w*(sigma_w*randn(3,1));
    sensor2_P1 = eye(n);
    % 第二个物体初始化
    s2 = [0;1.5;1;1;0;1.5];
    sensor1_x2 = s2 + f_w*(sigma_w*randn(3,1));
    sensor1_P2 = eye(n);
    sensor2_x2 = s2 + f_w*(sigma_w*randn(3,1));
    sensor2_P2 = eye(n);
    % 第三个物体初始化
    s3 = [0;1.5;0;1.5;1;1];
    sensor1_x3 = s3 + f_w*(sigma_w*randn(3,1));
    sensor1_P3 = eye(n);
    sensor2_x3 = s3 + f_w*(sigma_w*randn(3,1));
    sensor2_P3 = eye(n);
    for k = 1:N
        % 三个物体的真实局部轨迹
        s1 = f_x(s1) + f_w*(sigma_w*randn(3,1));
        s2 = f_x(s2) + f_w*(sigma_w*randn(3,1));
        s3 = f_x(s3) + f_w*(sigma_w*randn(3,1));
        true_sV(1,iter,k,:) = s1;
        true_sV(2,iter,k,:) = s2;
        true_sV(3,iter,k,:) = s3;
        
        % 传感器1的局部航迹估计
        sensor1_z1 = h_z(s1) + sigma_v*randn(3,1);
        sensor1_z2 = h_z(s2) + sigma_v*randn(3,1);
        sensor1_z3 = h_z(s3) + sigma_v*randn(3,1);
        
        % 1.传感器1对三个物体的观测
        sensor1_zV(1,iter,k,:) = sensor1_z1;
        sensor1_zV(2,iter,k,:) = sensor1_z2;
        sensor1_zV(3,iter,k,:) = sensor1_z3;
        
        % 2.对三个物体的prediction
        sensor1_p1 = f_x(sensor1_x1);
        sensor1_A1 = f_x_j(sensor1_x1);
        sensor1_pV(1,iter,k,:) = sensor1_p1;
        sensor1_P1 = sensor1_A1*sensor1_P1*sensor1_A1'+Q;
        
        sensor1_p2 = f_x(sensor1_x2);
        sensor1_A2 = f_x_j(sensor1_x2);
        sensor1_pV(2,iter,k,:) = sensor1_p2;
        sensor1_P2 = sensor1_A2*sensor1_P2*sensor1_A2'+Q;
        
        sensor1_p3 = f_x(sensor1_x3);
        sensor1_A3 = f_x_j(sensor1_x3);
        sensor1_pV(3,iter,k,:) = sensor1_p3;
        sensor1_P3 = sensor1_A3*sensor1_P3*sensor1_A3'+Q;
        % 3.对三个物体的update
        sensor1_y1 = h_z(sensor1_p1);
        sensor1_H1 = h_z_j(sensor1_p1);
        sensor1_K1 = sensor1_P1*sensor1_H1'*inv(sensor1_H1*sensor1_P1*sensor1_H1'+R);
        sensor1_x1 = sensor1_p1 + sensor1_K1*(sensor1_z1-sensor1_y1);
        sensor1_P1 = sensor1_P1-sensor1_K1*sensor1_H1*sensor1_P1;
        sensor1_xV(1,iter,k,:) = sensor1_x1;
        sensor1_RMSE(1,iter,k,1) = sqrt(sum((s1 - sensor1_x1).*(s1 - sensor1_x1)));
        
        sensor1_y2 = h_z(sensor1_p2);
        sensor1_H2 = h_z_j(sensor1_p2);
        sensor1_K2 = sensor1_P2*sensor1_H2'*inv(sensor1_H2*sensor1_P2*sensor1_H2'+R);
        sensor1_x2 = sensor1_p2 + sensor1_K2*(sensor1_z2-sensor1_y2);
        sensor1_P2 = sensor1_P2-sensor1_K2*sensor1_H2*sensor1_P2;
        sensor1_xV(2,iter,k,:) = sensor1_x2;
        sensor1_RMSE(2,iter,k,1) = sqrt(sum((s2 - sensor1_x2).*(s2 - sensor1_x2)));
        
        sensor1_y3 = h_z(sensor1_p3);
        sensor1_H3 = h_z_j(sensor1_p3);
        sensor1_K3 = sensor1_P3*sensor1_H3'*inv(sensor1_H3*sensor1_P3*sensor1_H3'+R);
        sensor1_x3 = sensor1_p3 + sensor1_K3*(sensor1_z3-sensor1_y3);
        sensor1_P3 = sensor1_P3-sensor1_K3*sensor1_H3*sensor1_P3;
        sensor1_xV(3,iter,k,:) = sensor1_x3;
        sensor1_RMSE(3,iter,k,1) = sqrt(sum((s3 - sensor1_x3).*(s1 - sensor1_x3)));
        
        % 传感器2的局部航迹估计
        sensor2_z1 = h_z(s1) + sigma_v*randn(3,1);
        sensor2_z2 = h_z(s2) + sigma_v*randn(3,1);
        sensor2_z3 = h_z(s3) + sigma_v*randn(3,1);
        
        % 1.传感器2对三个物体的观测
        sensor2_zV(1,iter,k,:) = sensor2_z1;
        sensor2_zV(2,iter,k,:) = sensor2_z2;
        sensor2_zV(3,iter,k,:) = sensor2_z3;
        
        % 2.对三个物体的prediction
        sensor2_p1 = f_x(sensor2_x1);
        sensor2_A1 = f_x_j(sensor2_x1);
        sensor2_pV(1,iter,k,:) = sensor2_p1;
        sensor2_P1 = sensor2_A1*sensor2_P1*sensor2_A1'+Q;
        
        sensor2_p2 = f_x(sensor2_x2);
        sensor2_A2 = f_x_j(sensor2_x2);
        sensor2_pV(2,iter,k,:) = sensor2_p2;
        sensor2_P2 = sensor2_A2*sensor2_P2*sensor2_A2'+Q;
        
        sensor2_p3 = f_x(sensor2_x3);
        sensor2_A3 = f_x_j(sensor2_x3);
        sensor2_pV(3,iter,k,:) = sensor2_p3;
        sensor2_P3 = sensor2_A3*sensor2_P3*sensor2_A3'+Q;
        % 3.对三个物体的update
        sensor2_y1 = h_z(sensor2_p1);
        sensor2_H1 = h_z_j(sensor2_p1);
        sensor2_K1 = sensor2_P1*sensor2_H1'*inv(sensor2_H1*sensor2_P1*sensor2_H1'+R);
        sensor2_x1 = sensor2_p1 + sensor2_K1*(sensor2_z1-sensor2_y1);
        sensor2_P1 = sensor2_P1-sensor2_K1*sensor2_H1*sensor2_P1;
        sensor2_xV(1,iter,k,:) = sensor2_x1;
        sensor2_RMSE(1,iter,k,1) = sqrt(sum((s1 - sensor2_x1).*(s1 - sensor2_x1)));
        
        sensor2_y2 = h_z(sensor2_p2);
        sensor2_H2 = h_z_j(sensor2_p2);
        sensor2_K2 = sensor2_P2*sensor2_H2'*inv(sensor2_H2*sensor2_P2*sensor2_H2'+R);
        sensor2_x2 = sensor2_p2 + sensor2_K2*(sensor2_z2-sensor2_y2);
        sensor2_P2 = sensor2_P2-sensor2_K2*sensor2_H2*sensor2_P2;
        sensor2_xV(2,iter,k,:) = sensor2_x2;
        sensor2_RMSE(2,iter,k,1) = sqrt(sum((s2 - sensor2_x2).*(s2 - sensor2_x2)));
        
        sensor2_y3 = h_z(sensor2_p3);
        sensor2_H3 = h_z_j(sensor2_p3);
        sensor2_K3 = sensor2_P3*sensor2_H3'*inv(sensor2_H3*sensor2_P3*sensor2_H3'+R);
        sensor2_x3 = sensor2_p3 + sensor2_K3*(sensor2_z3-sensor2_y3);
        sensor2_P3 = sensor2_P3-sensor2_K3*sensor2_H3*sensor2_P3;
        sensor2_xV(3,iter,k,:) = sensor2_x3;
        sensor2_RMSE(3,iter,k,1) = sqrt(sum((s3 - sensor2_x3).*(s1 - sensor2_x3)));
        
        % 局部航迹关联与融合W
        
    end
end
