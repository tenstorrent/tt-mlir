// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
    func.func @test_permute_mean_commute_downwards(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1152x1x1xbf16> {
        // CHECK: %[[REDUCE:[0-9]+]] = "ttir.mean"(%arg0
        // CHECK-SAME: dim_arg = [1 : i32, 2 : i32]
        // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[REDUCE]]
        // CHECK: return %[[PERMUTE]]
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %3 = "ttir.mean"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        return %3 : tensor<12x1152x1x1xbf16>
    }
    // Commute when reduce has keepdim = false is not currently supported
    func.func @test_permute_mean_keepdim_false_not_commute(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1152xbf16> {
        // CHECK: "ttir.permute"
        // CHECK: "ttir.mean"
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152xbf16>
        %3 = "ttir.mean"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = false}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152xbf16>) -> tensor<12x1152xbf16>
        return %3 : tensor<12x1152xbf16>
    }
    // If reduce op has multiple users, they all use same permuted input
    func.func @test_permute_mean_multiple_users_downwards(%arg0: tensor<12x7x7x1152xbf16>) -> (tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>) {
        // CHECK: %[[MEAN:[0-9]+]] = "ttir.mean"(%arg0,
        // CHECK-SAME: dim_arg = [1 : i32, 2 : i32]
        // CHECK: "ttir.permute"(%[[MEAN]]
        // CHECK-NOT: "ttir.permute"
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %3 = "ttir.mean"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        %4 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %5 = "ttir.add"(%3, %3, %4) : (tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        %6 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %7 = "ttir.multiply"(%3, %3, %6) : (tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        %8 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %9 = "ttir.relu"(%3, %8) : (tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        return %5, %7, %9 : tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>, tensor<12x1152x1x1xbf16>
    }
    // If permute has multiple users, it is not commuted downwards
    func.func @test_permute_multiple_users_mean_downwards(%arg0: tensor<12x7x7x1152xbf16>) -> (tensor<12x1152x1x1xbf16>, tensor<12x1152x7x7xbf16>) {
        // CHECK: "ttir.permute"
        // CHECK: "ttir.mean"
        // CHECK: "ttir.relu"
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %3 = "ttir.mean"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        %4 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %5 = "ttir.relu"(%1, %4) : (tensor<12x1152x7x7xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        return %3, %5 : tensor<12x1152x1x1xbf16>, tensor<12x1152x7x7xbf16>
    }

    func.func @test_permute_sum_inverse_permute_downwards(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1x1x1152xbf16> {
        // CHECK: "ttir.sum"{{.*}}dim_arg = [1 : i32, 2 : i32]{{.*}}
        // CHECK-NOT: "ttir.permute"
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %3 = "ttir.sum"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        %4 = ttir.empty() : tensor<12x1x1x1152xbf16>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x1xbf16>, tensor<12x1x1x1152xbf16>) -> tensor<12x1x1x1152xbf16>
        return %5 : tensor<12x1x1x1152xbf16>
    }
    func.func @test_permute_max_inverse_permute_downwards(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1x1x1152xbf16> {
        // CHECK: "ttir.max"{{.*}}dim_arg = [1 : i32, 2 : i32]{{.*}}
        // CHECK-NOT: "ttir.permute"
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %3 = "ttir.max"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        %4 = ttir.empty() : tensor<12x1x1x1152xbf16>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x1xbf16>, tensor<12x1x1x1152xbf16>) -> tensor<12x1x1x1152xbf16>
        return %5 : tensor<12x1x1x1152xbf16>
    }
    func.func @test_permute_min_inverse_permute_downwards(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1x1x1152xbf16> {
        // CHECK: "ttir.min"{{.*}}dim_arg = [1 : i32, 2 : i32]{{.*}}
        // CHECK-NOT: "ttir.permute"
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %3 = "ttir.min"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        %4 = ttir.empty() : tensor<12x1x1x1152xbf16>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x1xbf16>, tensor<12x1x1x1152xbf16>) -> tensor<12x1x1x1152xbf16>
        return %5 : tensor<12x1x1x1152xbf16>
    }
    func.func @test_permute_prod_inverse_permute_downwards(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1x1x1152xbf16> {
        // CHECK: "ttir.prod"{{.*}}dim_arg = [1 : i32, 2 : i32]{{.*}}
        // CHECK-NOT: "ttir.permute"
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x1xbf16>
        %3 = "ttir.prod"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x1xbf16>) -> tensor<12x1152x1x1xbf16>
        %4 = ttir.empty() : tensor<12x1x1x1152xbf16>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x1xbf16>, tensor<12x1x1x1152xbf16>) -> tensor<12x1x1x1152xbf16>
        return %5 : tensor<12x1x1x1152xbf16>
    }
    func.func @test_permute_reduce_and_inverse_permute_downwards(%arg0: tensor<12x7x7x1152xi1>) -> tensor<12x1x1x1152xi1> {
        // CHECK: "ttir.reduce_and"{{.*}}dim_arg = [1 : i32, 2 : i32]{{.*}}
        // CHECK-NOT: "ttir.permute"
        %0 = ttir.empty() : tensor<12x1152x7x7xi1>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xi1>, tensor<12x1152x7x7xi1>) -> tensor<12x1152x7x7xi1>
        %2 = ttir.empty() : tensor<12x1152x1x1xi1>
        %3 = "ttir.reduce_and"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xi1>, tensor<12x1152x1x1xi1>) -> tensor<12x1152x1x1xi1>
        %4 = ttir.empty() : tensor<12x1x1x1152xi1>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x1xi1>, tensor<12x1x1x1152xi1>) -> tensor<12x1x1x1152xi1>
        return %5 : tensor<12x1x1x1152xi1>
    }
    func.func @test_permute_reduce_or_inverse_permute_downwards(%arg0: tensor<12x7x7x1152xi1>) -> tensor<12x1x1x1152xi1> {
        // CHECK: "ttir.reduce_or"{{.*}}dim_arg = [1 : i32, 2 : i32]{{.*}}
        // CHECK-NOT: "ttir.permute"
        %0 = ttir.empty() : tensor<12x1152x7x7xi1>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xi1>, tensor<12x1152x7x7xi1>) -> tensor<12x1152x7x7xi1>
        %2 = ttir.empty() : tensor<12x1152x1x1xi1>
        %3 = "ttir.reduce_or"(%1, %2) <{dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<12x1152x7x7xi1>, tensor<12x1152x1x1xi1>) -> tensor<12x1152x1x1xi1>
        %4 = ttir.empty() : tensor<12x1x1x1152xi1>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x1xi1>, tensor<12x1x1x1152xi1>) -> tensor<12x1x1x1152xi1>
        return %5 : tensor<12x1x1x1152xi1>
    }
    func.func @test_permute_argmax_inverse_permute_downwards(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1x7x1152xi32> {
        // CHECK: "ttir.argmax"{{.*}}dim_arg = [1 : i32]{{.*}}
        // CHECK-NOT: "ttir.permute"
        %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
        %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
        %2 = ttir.empty() : tensor<12x1152x1x7xi32>
        %3 = "ttir.argmax"(%1, %2) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x7xi32>) -> tensor<12x1152x1x7xi32>
        %4 = ttir.empty() : tensor<12x1x7x1152xi32>
        %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x7xi32>, tensor<12x1x7x1152xi32>) -> tensor<12x1x7x1152xi32>
        return %5 : tensor<12x1x7x1152xi32>
    }
}
