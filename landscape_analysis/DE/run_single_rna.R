library(limma)
library(foreach)
library(doParallel)
require(svMisc)

wes = read.table("../../data/genomic/WES_variants_processed.csv.gz", stringsAsFactors = F, sep = ",", header = T, check.names = F)
mut_counts = data.frame(cbind(colnames(wes[,-1]), colSums(wes[,-1])), stringsAsFactors = F)
colnames(mut_counts) = c("Gene", "mut_count")
genes = mut_counts[mut_counts$mut_count>10,"Gene"]

rna = read.table("../../data/rna/rna_processed.csv", sep=",", header = T, stringsAsFactors = F, row.names = 1, check.names = F)
rna = rna[, colnames(rna) %in% wes$Cell_line]

genes = intersect(rownames(rna), genes)

# genes = c("TP53", "TTN")
cores=detectCores()
cl <- makeCluster(cores[1]-2) #not to overload your computer
registerDoParallel(cl)

res.df <- foreach(i = 1:length(genes),
                  .packages = c('limma'), .combine=rbind) %dopar% {
  gene = genes[i]
  
  cell_lines_wt = wes[wes[gene]==0, "Cell_line"]
  cell_lines_mut = wes[wes[gene]>0,"Cell_line"]
  rna_tmp = data.frame(rna[rownames(rna)==gene,], check.names = F)
  mutant = as.numeric(colnames(rna_tmp) %in% cell_lines_mut)
  
  design <- model.matrix(~1 + mutant)
  
  fit <- lmFit(rna_tmp, design)
  fit <- eBayes(fit)
  rna_res = topTable(fit, coef=ncol(design), number = nrow(rna_tmp))
  rownames(rna_res) = c(gene)
  return(rna_res)
}
#stop cluster
stopCluster(cl)

write.table(res.df, "../../result_files/de/rna_signle.csv", sep = ",", quote = F)
