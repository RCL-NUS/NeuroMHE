%plotting
subplot(3,1,1);
plot(e_resi_f(:,1));
hold on;
plot(e_resi_f_updated(:,1));
legend('m=0.752kg','m=0.772kg','NumColumns',2,'Location','northoutside');
ylabel('$\delta f_x$ [N]','Interpreter','latex');
grid on;

subplot(3,1,2);
plot(e_resi_f(:,2));
hold on;
plot(e_resi_f_updated(:,2));
ylabel('$\delta f_y$ [N]','Interpreter','latex');
grid on;

subplot(3,1,3);
plot(e_resi_f(:,3));
hold on;
plot(e_resi_f_updated(:,3));
grid on;
xlabel('data points')
ylabel('$\delta f_z$ [N]','Interpreter','latex');
grid on;