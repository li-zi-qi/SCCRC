clear
clc
close all

load('..\EYaleB_32x32.mat')

ClassNum = length(unique(gnd));
EachClassNum = zeros(1,ClassNum);
for i=1:ClassNum
    EachClassNum(i) = sum(i==gnd);
end

s_lambda = [1e-4 1e-5 1e-5 1e-4 1e-4 1e-4];
s_lambda1 = [1e-5 1e-4 1e-4 1e-3 1e-4 1e-3];

k = 1;
for train_num = 5:5:30
    lambda = s_lambda(k);
    lambda1 = s_lambda1(k);
    
    train_ind = [];
    for ii=1:ClassNum
        temp = zeros(1,EachClassNum(ii));
        temp(1:train_num) = 1;
        train_ind = [train_ind,temp];
    end
    
    train_ind = logical(train_ind);
    test_ind = ~train_ind;
    
    train_data = fea(:,train_ind);
    train_label = gnd(:,train_ind);
    
    test_data = fea(:,test_ind);
    test_label = gnd(:,test_ind);
    
    train_tol = length(train_label);
    test_tol = length(test_label);
    
    %��λ��
    train_norm=normc(train_data);
    test_norm=normc(test_data);
    
    X = train_norm;
    tr_sym_mat = zeros(train_tol);
    for ci = 1 : ClassNum
        ind_ci = find(train_label == ci);
        tr_descr_bar = zeros(size(X));
        tr_descr_bar(:,ind_ci) = X(:, ind_ci);
        tr_sym_mat = tr_sym_mat + tr_descr_bar' * tr_descr_bar;
    end
    
    P = (1+lambda1)*(X'*X+lambda*eye(train_tol)+lambda1*tr_sym_mat)\X';
    
    pre_label=zeros(1,test_tol);
    %     [~,C] = NRC(train_norm, test_norm);
    % h = waitbar(0,'Please wait...');
    parfor ii=1:test_tol
        y=test_norm(:,ii);
        
        %         src_coeff = l1_ls_nonneg(train_norm,y,src_lambda,[],true);
        src_coeff = SolveFISTA(train_norm, y);
        %     xp = SolveHomotopy(train_norm, y, 'lambda', src_lambda, 'tolerance', 1e-5, 'stoppingcriterion', 3);
        
        %     xp = SolveHomotopy_CBM_std(train_norm, y,'lambda', src_lambda);
        
        %     xp = SolveDALM(train_norm,y, 'lambda',src_lambda,'tolerance',1e-3);
        
        ccrc_coeff = P*y;
        xp = src_coeff.*ccrc_coeff;
        %����sparse���󣬴�СΪtrain_tol*ClassNum�������train_tol������ֵ
        W=sparse([],[],[],train_tol,ClassNum,train_tol);
        
        %�õ�ÿ���Ӧ��ϵ��?
        for jj=1:ClassNum
            ind=(jj==train_label);
            W(ind,jj)=xp(ind);
        end
        
        %����������ÿ���ع���֮��Ĳв�?
        temp=train_norm*W-repmat(y,1,ClassNum);
        residual=sqrt(sum(temp.^2));
        
        [~,index]=min(residual);
        pre_label(ii)=index;
        
        %     % computations take place here
        %     per = i / test_tol;
        %     waitbar(per, h ,sprintf('%2.0f%%',per*100))
    end
    % close(h)
    accuracy = sum(pre_label==test_label)/test_tol
    k = k+1;
end

