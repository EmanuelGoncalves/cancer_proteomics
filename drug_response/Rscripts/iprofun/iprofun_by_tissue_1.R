library(iProFun)

sanger_cna = read.table("./data/iprofun/cna.tsv", sep = "\t", check.names = F, header = T)
sanger_rna = read.table("./data/iprofun/rna.tsv", sep = "\t", check.names = F, header = T)
sanger_methy = read.table("./data/iprofun/methy.tsv", sep = "\t", check.names = F, header = T)
sanger_protein = read.table("./data/iprofun/protein.tsv", sep = "\t", check.names = F, header = T)

sanger_rna_pca = read.table("./data/iprofun/rna_pca.tsv", sep = "\t", check.names = F, header = T)
sanger_protein_pca = read.table("./data/iprofun/protein_pca.tsv", sep = "\t", check.names = F, header = T)

ic50 = read.table("./data/drug/ic50_processed_median.csv", sep = ",", check.names = F, header = T, stringsAsFactors = F)
cellline = ic50[,c("Cell line name", "Tissue")]
cellline = cellline[!duplicated(cellline),]

for (tissue in unique(cellline$Tissue)){
  print(paste0("Runing ", tissue))
  tissue_cellline = cellline[cellline$Tissue==tissue, "Cell line name"]
  cols = c("GENE_ID", tissue_cellline)
  sanger_cna_tissue = sanger_cna[, colnames(sanger_cna) %in% cols]
  sanger_rna_tissue = sanger_rna[, colnames(sanger_rna) %in% cols]
  sanger_methy_tissue = sanger_methy[, colnames(sanger_methy) %in% cols]
  sanger_protein_tissue = sanger_protein[, colnames(sanger_protein) %in% cols]
  sanger_rna_pca_tissue = sanger_rna_pca[, colnames(sanger_rna_pca) %in% cols]
  sanger_protein_pca_tissue = sanger_protein_pca[, colnames(sanger_protein_pca) %in% cols]
  
  iprofun_permutate_result <- iProFun_permutate(ylist = list(sanger_rna_tissue, sanger_protein_tissue), 
                                                xlist = list(sanger_cna_tissue, sanger_methy_tissue), 
                                                covariates = list(sanger_rna_pca_tissue, sanger_protein_pca_tissue), 
                                                sub.ID.common=NULL, colum.to.keep=c("GENE_ID"), pi = rep(0.05, 3), 
                                                permutate_number = 10, fdr = 0.1, PostCut = 0.99, filter <- c(1,0), 
                                                grids = c(seq(0.75, 0.99, 0.01), seq(0.991, 0.999, 0.001), seq(0.9991, 0.9999, 0.0001)), 
                                                seed=123)
  filename = paste0("./data/iprofun/iprofun_permutate_result_", tissue, "_10.rds")
  saveRDS(iprofun_permutate_result, filename)
}




