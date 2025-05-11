import numpy as np

def SNR_convert(SNR) -> float:
    
    return 10 ** (-SNR / 10)

def get_precoder_matrix(H_ideal, precoder_rank) -> np.array:
    # H_ideal : [NTx, NRx]
    U, S, Vt = np.linalg.svd(H_ideal)
    W = Vt[:, 0:precoder_rank].conj() / np.sqrt(precoder_rank)
    return W

# Фугнкция нахождения пропускной способности OFDM сигнала в MIMO системе
def get_Capacity(H_ideal: np.array, H_restored: np.array, precoder_rank, SNR_dB, channel_shape) -> np.array:
    # H : [NTx, NRx, NSc, NTTi, NUe]
    P_noise = SNR_convert(SNR_dB)

    NTx = channel_shape[0]
    NRx = channel_shape[1]
    NSc = channel_shape[2]
    NTTi = channel_shape[3]
    NUe = channel_shape[4]

    
    if NUe == 1:
        mean_capacity = np.zeros(NTTi) # for each user and TTi
        for n_tti in range(NTTi):
            Capacity = 0
            for n_sc in range(NSc):        
                # W = np.linalg.svd(H[:, :, n_sc, n_tti, n_ue])
                W = get_precoder_matrix(H_ideal[:, :, n_sc, n_tti].T, precoder_rank)
                H_prec = H_restored[:, :, n_sc, n_tti].T @ W
                corr_matrix = H_prec.conj().T @ H_prec
                # print(corr_matrix)
                # print()
                Capacity += np.real(np.log2(np.linalg.det(np.eye(precoder_rank) + 1 / P_noise * corr_matrix)))
            Capacity /= NSc
            mean_capacity[n_tti] = Capacity

    else:
        mean_capacity = np.zeros((NUe, NTTi)) # for each user and TTi
        for n_ue in range(NUe):
            for n_tti in range(NTTi):
                Capacity = 0
                for n_sc in range(NSc):
                    # W = np.linalg.svd(H[:, :, n_sc, n_tti, n_ue])
                    W = get_precoder_matrix(H_ideal[:, :, n_sc, n_tti, n_ue], precoder_rank) / np.sqrt(precoder_rank)
                    H_prec = H_restored[:, :, n_sc, n_tti, n_ue].reshape(NRx, NTx) @ W
                    corr_matrix = H_prec.conj().T @ H_prec  # diag matrix
                    Spef = np.real(np.log2(np.linalg.det(np.eye(precoder_rank) + 1 / P_noise * corr_matrix)))
                    Capacity += Spef
                Capacity /= NSc
                mean_capacity[n_ue, n_tti] = Capacity

    return mean_capacity

def cdf(data) -> list:
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    return (sorted_data, cdf)