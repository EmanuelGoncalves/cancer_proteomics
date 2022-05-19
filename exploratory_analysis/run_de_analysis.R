library(limma)
library(foreach)
library(doParallel)
require(svMisc)

wes = read.table(
  "./data/genomic/WES_variants_processed.csv.gz",
  stringsAsFactors = F,
  sep = ",",
  header = T,
  check.names = F
)
mut_counts = data.frame(cbind(colnames(wes[, -1]), colSums(wes[, -1])), stringsAsFactors = F)
colnames(mut_counts) = c("Gene", "mut_count")
genes = mut_counts[mut_counts$mut_count > 10, "Gene"]


protein = read.table(
  "./data/protein/protein_de_processed_20211005.csv",
  stringsAsFactors = F,
  sep = ",",
  header = T,
  row.names = 1,
  check.names = F
)
protein = protein[, colnames(protein) %in% wes$Cell_line]
genes = intersect(rownames(protein), genes)

cores = detectCores()
cl <- makeCluster(10) #not to overload your computer
registerDoParallel(cl)

res.df <- foreach(i = 1:length(genes),
        .packages = c('limma'), .combine=rbind) %dopar% {
          gene = genes[i]

          cell_lines_wt = wes[wes[gene] == 0, "Cell_line"]
          cell_lines_mut = wes[wes[gene] > 0, "Cell_line"]
          protein_tmp = data.frame(protein[rownames(protein)==gene,], check.names = F)

          mutant = as.numeric(colnames(protein_tmp) %in% cell_lines_mut)

          design <- model.matrix( ~ 1 + mutant)

          fit <- lmFit(protein_tmp, design)
          fit <- eBayes(fit)
          protein_res = topTable(fit, coef = ncol(design), number = nrow(protein_tmp))
          rownames(protein_res) = c(gene)
          return(protein_res)
        }
stopCluster(cl)

write.table(res.df, "./result_files/de/protein_signle.csv", sep = ",", quote = F)