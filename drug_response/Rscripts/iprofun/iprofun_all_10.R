library(iProFun)

sanger_cna = read.csv("../data/iprofun/cna.tsv", sep = "\t")
sanger_rna = read.csv("../data/iprofun/rna.tsv", sep = "\t")
sanger_methy = read.csv("../data/iprofun/methy.tsv", sep = "\t")
sanger_protein = read.csv("../data/iprofun/protein.tsv", sep = "\t")

sanger_rna_pca = read.csv("../data/iprofun/rna_pca.tsv", sep = "\t")
sanger_protein_pca = read.csv("../data/iprofun/protein_pca.tsv", sep = "\t")

iprofun_permutate_result <- iProFun_permutate(ylist = list(sanger_rna, sanger_protein), 
                                              xlist = list(sanger_cna, sanger_methy), 
                                              covariates = list(sanger_rna_pca, sanger_protein_pca), 
                                              sub.ID.common=NULL, colum.to.keep=c("GENE_ID"), pi = rep(0.05, 3), 
                                              permutate_number = 10, fdr = 0.1, PostCut = 0.99, filter <- c(1,0), 
                                              grids = c(seq(0.75, 0.99, 0.01), seq(0.991, 0.999, 0.001), seq(0.9991, 0.9999, 0.0001)), 
                                              seed=123)

saveRDS(iprofun_permutate_result, "../data/iprofun/iprofun_permutate_result_10.rds")
