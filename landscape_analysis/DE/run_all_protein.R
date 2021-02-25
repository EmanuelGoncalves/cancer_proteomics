library(limma)
library(foreach)
library(doParallel)
require(svMisc)

wes = read.table(
  "../../data/genomic/WES_variants_processed.csv.gz",
  stringsAsFactors = F,
  sep = ",",
  header = T,
  check.names = F
)
mut_counts = data.frame(cbind(colnames(wes[, -1]), colSums(wes[, -1])), stringsAsFactors = F)
colnames(mut_counts) = c("Gene", "mut_count")
genes = mut_counts[mut_counts$mut_count > 10, "Gene"]

protein = read.table(
  "../../data/protein/protein_de_processed.csv",
  stringsAsFactors = F,
  sep = ",",
  header = T,
  row.names = 1,
  check.names = F
)
protein = protein[, colnames(protein) %in% wes$Cell_line]

protein_knn = read.table(
  "../../data/protein/protein_de_processed_knn.csv",
  stringsAsFactors = F,
  sep = ",",
  header = T,
  row.names = 1,
  check.names = F
)
protein_knn = protein_knn[, colnames(protein_knn) %in% wes$Cell_line]

protein_impute_q1 = data.frame(protein, check.names = F)
protein_impute_q1[is.na(protein_impute_q1)] = -2.242616

# genes = c("TP53", "TTN")
cores = detectCores()
cl <- makeCluster(10) #not to overload your computer
registerDoParallel(cl)

foreach(i = 1:length(genes),
        .packages = c('limma')) %dopar% {
          gene = genes[i]
          
          cell_lines_wt = wes[wes[gene] == 0, "Cell_line"]
          cell_lines_mut = wes[wes[gene] > 0, "Cell_line"]
          
          mutant = as.numeric(colnames(protein) %in% cell_lines_mut)
          
          design <- model.matrix( ~ 1 + mutant)
          
          fit <- lmFit(protein, design)
          fit <- eBayes(fit)
          protein_res = topTable(fit, coef = ncol(design), number = nrow(protein))
          if ((protein_res[rownames(protein_res) == gene, "adj.P.Val"] < 0.05) &
              (protein_res[rownames(protein_res) == gene, "logFC"] > 0.5)) {
            print(paste0("writing for ", gene))
            flush.console()
            write.table(
              protein_res,
              paste0(
                "../../result_files/de/protein_ex/",
                gene,
                "_protein.csv"
              ),
              sep = ",",
              quote = F
            )
          }
        }
#stop cluster
stopCluster(cl)

# for (i in 1:length(genes)){
#   progress(i/length(genes))
#   gene = genes[i]
#
#   cell_lines_wt = wes[wes[gene]==0, "Cell_line"]
#   cell_lines_mut = wes[wes[gene]>0,"Cell_line"]
#
#   mutant = as.numeric(colnames(protein_impute_q1) %in% cell_lines_mut)
#   if (sum(mutant)==0) {
#     next
#   }
#   design <- model.matrix(~1 + mutant)
#
#   fit <- lmFit(protein_impute_q1, design)
#   fit <- eBayes(fit)
#   protein_res = topTable(fit, coef=ncol(design), number = nrow(protein_impute_q1))
#   if (gene %in% rownames(protein_res)[1:10]){
#     print(paste0("writing for ", gene))
#     flush.console()
#     write.table(protein_res, paste0("../../result_files/de/", gene,"_protein.csv"), sep = ",", quote = F)
#   }
# }
#
