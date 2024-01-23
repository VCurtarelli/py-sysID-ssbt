tic

parfor k = 1:32
    Xp_k = Xp(:, :, k);
    [U, S, V] = svd(Xp_k);
    n = rank(Xp_k);
    U_ = U(:, 1:n);
    S_ = S(1:n);
    V_ = V(:, 1:n);
    A = V_' / diag(S_) * U_';
end

toc