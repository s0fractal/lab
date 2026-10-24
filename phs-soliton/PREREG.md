# PREREG v1: Pure-PHS vs Soliton-PHS vs Control-GRU on selective recall and a modular automaton

> Generated from `prereg.json` by `tools/prereg.py write`; `tools/prereg.py check` refuses a differing file. Version 1, tag `prereg-v1`, written 2026-10-24.

## Status

- Frozen at tag prereg-v1. A change after the tag is a new PREREG version with its reason; results of this version stay in the report.

## Hypothesis

Physically structured soliton memory helps: Soliton-PHS (a port-Hamiltonian reservoir on the cotangent torus plus a Frenkel-Kontorova ring with periodic pinning) has a higher noise threshold sigma_crit and a slower decline of accuracy with T on task A than Pure-PHS (the same reservoir without the lattice) and than Control-GRU of equal state dimension, and is not worse than Pure-PHS by more than 0.05 mean accuracy on task B.

## Sources

- starter PDF: starter/port-hamiltonian-soliton-reservoirs.pdf, SHA-256 0789a82d56a049e94ee0f838e7a5ef99fc6c74007ddd52b5a102c9758d6cf1b1 (6 pages; bottoms of pages 1, 5 and 6 clipped)
- transcription: starter/port-hamiltonian-soliton-reservoirs.md with transcription notes 1-10
- starter checks: starter/check_starter.py: 12/12 PASS; byte-identical to check_starter.out on arm64 macOS, last digits differ on x86-64 Linux
- goal: GOAL.md: decisions 1-7, the reconstructed verdict rule, stages S0-S7
- calibration: calibration/trials.jsonl (every job), calibration/SUMMARY.md, calibration/probes.jsonl

## Environment

- python: `3.13.15`
- torch: `2.14.1`
- numpy: `2.5.3`
- matplotlib: `3.11.2`
- lock: `uv.lock (Linux resolves torch from the PyTorch CPU index)`
- machine: `Apple M4 Pro, 14 cores (10 performance + 4 efficiency), 48 GB, macOS (Darwin 27.0.0)`
- device: `cpu, float32, torch.use_deterministic_algorithms(True), 1 thread per run`
- determinism: `a run is bitwise reproducible from its seed (checked: identical curves, validation, cells and physics on two runs under parallel load)`

## Models

| kind | model | role | calibration trial | params (A) | params (B) | config |
|---|---|---|---|---|---|---|
| pure | Pure-PHS | verdict | p07 | 3560 | 2534 | `{"N": 32, "S": 4, "d": 2, "dt": 1.0, "gamma": 0.5, "j_scale": 0.1, "kappa0": 1.0, "lr": 0.01, "u_max": 2.0}` |
| soliton | Soliton-PHS | verdict | s07 | 3562 | 2536 | `{"N": 32, "S": 4, "d": 2, "dt": 1.0, "g_lat0": 1.0, "g_pin0": 0.5, "gamma": 0.5, "j_scale": 0.1, "kappa0": 1.0, "lr": 0.01, "m_pin": 2, "u_max": 2.0}` |
| gru | Control-GRU | verdict | g06 | 54279 | 52101 | `{"N": 32, "d": 2, "lr": 0.007}` |
| gru_pm | Control-GRU-pm | reported, not in the verdict | q07 | 3479 | 2455 | `{"lr": 0.03}` |

## Optimisation (identical for every model)

`{"batch": 64, "clip": 1.0, "eval_every": 100, "n_val": 1024, "steps": 24000, "weight_decay": 0.01}`

## Tasks

- **A.** selective copy-recall: input tokens 0 (blank), 1..7 (symbols), 8 (query); marker y at t = 0, 1; t = 2..T-2 blanks with symbol distractors at p_dist = 0.25; query at t = T-1; cross-entropy on the last step only; 7 classes, chance 1/7.
- **B.** modular automaton q = 5: commands 0..3 uniform; s_{-1} = 0; s_t = (s_{t-1} + x_t) mod 5 for x_t < 3, s_t = (5 - s_{t-1}) mod 5 for x_t = 3; dense cross-entropy; Acc(T) on the final state.
- **training noise.** sigma drawn per sequence uniformly from {0, 0.05, ..., 0.35} (the PDF grid) for training and validation.
- **test cells.** 4096 sequences per (task, T, sigma) for sigma in the evaluation grid 0.0..2.0 step 0.05, each from its own seed, shared by every model and seed.

Test cells: 4096 sequences per (task, T, sigma), SHA-256 of `phs.tasks.digest`:

| cell | digest |
|---|---|
| A/16/0.00 | `a6190acb4fd333bbbd7d25451f918ca91e536600e137862a5e4ed5034a656bc6` |
| A/16/0.05 | `e773afd97ab719dd0f2fc2ba42d2a46c6ed11cdead1f918e5645136d63ad748a` |
| A/16/0.10 | `2be895de9501fb93e456d93627a6bb9d96362c7ea587a8e9cab72a2902f36cfe` |
| A/16/0.15 | `fb575d06403cb9a77e6ab590fe78549dceac984def8a9e4fd9ff4f7899c9b5fd` |
| A/16/0.20 | `f3775b5467a0910565e6deda9d71da99325a497aff1d37e73c99cbdf8b64b389` |
| A/16/0.25 | `e9d0b282b13a4edfc1de9ea44284dbb7a8b4efd8e8a654ec20dc30b503d7ed58` |
| A/16/0.30 | `b8f346a937cb0a139071fe15b5d2a5a1ad9e0f84982be7056d1726f660dfb798` |
| A/16/0.35 | `bd30e1ae28181b2f1653e044bda59ac7145326389ee8d9f485fc2afb1f7c3d78` |
| A/16/0.40 | `93e59ccf340d6e97042c803642a24ad4941b5095c6b9761d8ade5caecd58985b` |
| A/16/0.45 | `0122faa62447648f6085a7820fb400af55bf300abee2b58819b176db08a4c8ac` |
| A/16/0.50 | `cd6fb6aa96e89aa66db9575b675b3a6d972bf8077174e822c6e85dd9dd373467` |
| A/16/0.55 | `2744c124245e0c8c34bec2f771579cf6cf6c24969e6a558ed5edbca713d1edd8` |
| A/16/0.60 | `28cf6831338684cfe6ad2392e5b310919665d75644ca6041ee8442a5a9429c7e` |
| A/16/0.65 | `a8e00e6cdae1387f5d7afe48e7eea236899e55cabf48f3f0f798c04c7b7be820` |
| A/16/0.70 | `6483bef6e35365a4c4e28f8db629fd731a97d50874bad04a9e4a2738ff95f932` |
| A/16/0.75 | `493565fad50856cb2d6d75d47837f354af2198934bcc74c2668359a661dea077` |
| A/16/0.80 | `f4aa0548f61f640d5d099136bb2cf6dedefd278ecf91d4e1904f275a4ea5ed6d` |
| A/16/0.85 | `fdd57c5e826501860dbf0d7778d7aa8ce76e1241907a47084ca72a8c7758742a` |
| A/16/0.90 | `2fae628587afe3645094721d5c9eeb7947c36b8a1521a997061c3606b9e614ba` |
| A/16/0.95 | `de4a9c91bbeb6efe610003b3dbbd5217f82f39ec8c5cb80ae7defa7ff5ecfb19` |
| A/16/1.00 | `87ba820a54bb0bdea8a90ce02ef23448f88240bfa1cbb2b24423e0fbc9b509b0` |
| A/16/1.05 | `27d41beefca227f53964bf989cd253080aee41822e5da4376db672588370a7d8` |
| A/16/1.10 | `18875e8a93d60a795a2213088613c6d36ca47b1c829171922e960a2f418b3d80` |
| A/16/1.15 | `50cf17a53a5fec8e1f3b627fe353ef5dc1359914afdf96fc1387e9766332e942` |
| A/16/1.20 | `6f7ff6d333c1a794973bbeced958749355bdcf6cf2d5d0e24581f7410587fa2b` |
| A/16/1.25 | `7466bbe9687c2f6ae2ca28b346d0657600471ec19076146832270d2d8af06f30` |
| A/16/1.30 | `aafac483d78d0aea1a690502d53a3521f87688bc834ceb37090be126b3109b12` |
| A/16/1.35 | `5fc11e65dd6ba6fa6fe8389bb5d0d4aa97de46ca3819bb46398c7e9364976d8f` |
| A/16/1.40 | `24facc4562582b1b9b0aec0550f7ef76ef8006c541b2d1782c1ca517954195c6` |
| A/16/1.45 | `63515864a0c12ca192f4d02a781cd02db37c762ac80da53cda014ca045cb5a7a` |
| A/16/1.50 | `0cf7d4dd088b2944f9298395bce11f7385f785276cb3e15311f61b304a6928f2` |
| A/16/1.55 | `f7a963a45f28ff986d6195aa0cc025b7249becee4fa3cd885107d412348ca1e4` |
| A/16/1.60 | `1bf502a031bfe062b61a3255c98b66f49a0a20e0b103ecbe7ae7325661dd61a9` |
| A/16/1.65 | `77b533464218316c839bd7b7cc20d4c563b46b8d7f86e34b7ed1d0a79730cae1` |
| A/16/1.70 | `154829f0ce0d24fc1dfae1ce77277bdfc6c57e28e91125b4e9f12c7a33c170e4` |
| A/16/1.75 | `dd7ca5648b450ea4942e86127bee84908db2e2df7bb0e9f09b1d8f92543c16fd` |
| A/16/1.80 | `72fa3c965d50f1d76c3fd5b68ea9853e4ba3e243afa49a7174ef3fd475e01c20` |
| A/16/1.85 | `a9f7207f856770a1cd0e4d196267af049f543bb3f64c6d01a4f600346bd2ca91` |
| A/16/1.90 | `151b3419fade1cb986a602a9c8b79216f70537ea348c9318071e45f2e8b63c04` |
| A/16/1.95 | `686c25db986b85fcb4b2924ff9dcf863efdefcf053526e06fcae4d22cf92d2b2` |
| A/16/2.00 | `5fe1d39b31d12fb37b708189c5befafda764ba74d663dbf578d15e99fec9ecbf` |
| A/32/0.00 | `4cf3bada71e3b35aff53b72fcd2f0736e4b4d2f0790165f31904e22a83857ab3` |
| A/32/0.05 | `0414c9b87e892783514a8cd51dbb2c21f272ff0dc501aad840e6e1ee80a4a0ef` |
| A/32/0.10 | `bb846908e910dc492299dcc38969bf77c6aba5b9dd1b9e78d4c37a1bcb93f773` |
| A/32/0.15 | `99b1d7a379a5748e8ef18b117d0ead16992a8dbc06bf5ef625b0e95ca34b32ac` |
| A/32/0.20 | `e3769c6e1bf814c568d1f4fd7612125747764dcbabf4ab29c1a8e69ff6152d96` |
| A/32/0.25 | `0cde5779255a6656c31abb949818e072950509ae129a107feae9c96c2e9e0829` |
| A/32/0.30 | `db6d3f197815b6d96121b204c9820a0a06d066bc4d99f7cb465ac9af76aee8e5` |
| A/32/0.35 | `fa0b96f39e46a862e1b46bc3480609a681243298f687b437f7547e0ee001b30a` |
| A/32/0.40 | `318a8411b4f2a3940c2e3e44273517bdc5604fa01d4a17c07d442ea44510d0f6` |
| A/32/0.45 | `4d025c3b85379b8b49f22c706ed3fef994db131730f38a69b5e7183c2d67d2d4` |
| A/32/0.50 | `6373b8381f47089d5c59beeac4c0331c339006bdbeaff8ea9f4735eb73a37e69` |
| A/32/0.55 | `214c25a243e000ac0fd096bacaa714ef66d3bbf9480c7be02cc7f280c53616b2` |
| A/32/0.60 | `c9452c75598b9be3018989b91807b3ba8cc63076b12e0c3330adde7d08424752` |
| A/32/0.65 | `8c61207a37b97a98c397d8ae787a1d85553996eb55681728d089ad60e5cc6c57` |
| A/32/0.70 | `d726f26008bbf67b68b53bad38b7151f71c72ca20a2877b7e3b68db69f20d59c` |
| A/32/0.75 | `5ca19df0f964928bbf9f096bdb1d359af7c189e74d585360fa6313aa660e3e78` |
| A/32/0.80 | `62f6aae5175e8dc254bb9616d6f8cc9ab909b01f72b6a738c27e0b2c62c98cc8` |
| A/32/0.85 | `2afa58ce25d2b820260296456eb7f8f0a4f6ffd312d6e0671fb41c8534fc9a97` |
| A/32/0.90 | `f3e198a8efea1cdeb0f4d5b81c936d2df60d5f7448ea52b2fbaeaa42e4350779` |
| A/32/0.95 | `006bd17debbc29b6414c23d5db31f587b9f8d13d55e30debe7e48da2b25c4f5a` |
| A/32/1.00 | `a7a27c000622248e976fa4ad24841b836214d166913772d0d6fce518ef73da05` |
| A/32/1.05 | `55c77f0975f4982a38b8135ce684f136cc97af001d4af6e914008941864911d0` |
| A/32/1.10 | `34110487ab11b115281b412e751faec4712f013d5a7a2d4e5002a12686c1e57f` |
| A/32/1.15 | `549f95979841348a0c1154ca8c7f1ebd7bf94c772c4148e6dfdbba936ea4803b` |
| A/32/1.20 | `08712f8722bb1f1af53462e8c377cdfbc0a5c29f4e6f07b084c2576b987252a5` |
| A/32/1.25 | `3f401c800cc8d94f28a525445c4c4e6cc76f65f9e3b22f0e8bc0cbcddc029e70` |
| A/32/1.30 | `534497e4327d4ccce2cf883750b21c2301dce30b22294d7502793d795862d68d` |
| A/32/1.35 | `51a02017aa93e931fff99d7a55b4941139f635393907b524a15344922519f806` |
| A/32/1.40 | `86d49c91756db90b2990852526c01e8efcfba19714ed73b7867a9d2232dc5eec` |
| A/32/1.45 | `daca7557ce46128e3686ff9949ebae816d672895ab3cddd568c983de8d8a1e59` |
| A/32/1.50 | `8e9c94ff580c97c36d8c879f4c366617e23dc705f39992a2df5e57247fba2af1` |
| A/32/1.55 | `30a4d1792e88dd825a17cf2f868b8dd36e757a39567fc8aceba4a3bd53af68cc` |
| A/32/1.60 | `dd4c20fda402b1e9ade3ada515609278294c8c3e39eed59f2fb33c6899ef6d51` |
| A/32/1.65 | `9670eed5323a7007a030acd2e8f3d976cee38e25d32511eadbb12d7ba04ac98c` |
| A/32/1.70 | `9746ceffcd921b35e7a5a8a9cd41586d28406d164e9ebfc7f36460b083a0b1df` |
| A/32/1.75 | `c144df835900eb311dbb80450d30239d3d0ae08bbe8cebc09bf99a2452dc0981` |
| A/32/1.80 | `fb5cbc1b9b4d29cbcae3019cc575e097a7b03be74da9bf295979925d3214af38` |
| A/32/1.85 | `d4578eaec881d272d926e87b3de1a9e6aa7752b67b2dc1d53b60ac91f1a30d3b` |
| A/32/1.90 | `f3bddcd86a8637761f5407ae5e2eba3db918db21b05ac503cd7800d2db8ed0b4` |
| A/32/1.95 | `5ae772277c61edb78017fe0090230a41d9176cea03a5364fd6c0f2c9e5a16902` |
| A/32/2.00 | `10c9070a9a42233728fe85bac23356f7fa444fb57fb37d033224c18d342b79a3` |
| A/64/0.00 | `37c08de52c75d504d111b6dbd994ca9fd9a75ed557102695feec02d57c4a1b21` |
| A/64/0.05 | `a5909a5e498bf277a2b1eaddf792a8d572ef275bed8fd6e6d3ff1a5adf5ba84a` |
| A/64/0.10 | `ae251ce7385d62572496df6437c171371d6097de462dfb5aaa963632b8861fb6` |
| A/64/0.15 | `a5686c0349927e1d050186f9b2617122ba09895d75d2b04a0662e6c6753aa804` |
| A/64/0.20 | `f8eb91d9b7958b88aa174edb5ad40e58943560bf31022b1e053452170e518b2f` |
| A/64/0.25 | `03471064de9c493003cef97de07556a53ad191c66eac42834c9beed2a9f60fa6` |
| A/64/0.30 | `e4459057ae3f89c4e2123af0b907970ce8d697f48b51fb63cfb17b00a7654cd1` |
| A/64/0.35 | `47846198834c15a6fefea5ef9f12caf6b0718cd7479cd47f607b3bd648c150b4` |
| A/64/0.40 | `cad7155b80d6617deaff7db963407aabe9edf7e9b929b7326f902b900d81f244` |
| A/64/0.45 | `53efeedfc9670d8421e4cbdd5ab6fe0abccbed0b1b791088c63031253a7f5099` |
| A/64/0.50 | `809fc2b619fd6ebc3b59bd5113f03e9638af1e066b232d715aea151c8da25725` |
| A/64/0.55 | `2d1b64117670c8b07397d30773458c387f1f0ac9cefe06316813198c2bc8942d` |
| A/64/0.60 | `fc80179c2f2be1ffe2e61e0f221a45d553da14d5e34f6535548e6c4f9e077bc8` |
| A/64/0.65 | `58ded78838b345ad454b3876876ca73497b9d5c403d5bc71264104291c4bbcbb` |
| A/64/0.70 | `f075192f33efc352b738db86fe4d1a9ff47e4860bcf6db1df098459708ce3d95` |
| A/64/0.75 | `74af2edddbb2c283c9ecb97c44964dffe3b5e12a7289d3513414e9535dd97195` |
| A/64/0.80 | `a923f642d87d25ae7ca4d70f280fcf127ccd4d1028e4a2e203878c77b8a908fb` |
| A/64/0.85 | `fda95626dfa418c7c19cfecef9f2c6eb647a80c85faf7012bee81eff9ac4f876` |
| A/64/0.90 | `8945aa1e769470d8fd95ed4cb5c070ed5c4966bdde7b7c1c071f292945e85139` |
| A/64/0.95 | `875f9c45ae6aebaa78ca65794b99f3aff5d0ef3d4e353617507d47cfb37ca318` |
| A/64/1.00 | `8a134f4566efe14f26b0625581872cf732a3c6de268d74357bb67d45468336b8` |
| A/64/1.05 | `919d47c96f78ebd7f32d8301770981d70a9c1d2dbed83dbf6271f53701718d84` |
| A/64/1.10 | `fe779bbe3acbaa0310f248a392ae6e1965345ab44d73c831bc9bd5c82b2d4dc1` |
| A/64/1.15 | `7bcac3f978b274cc9c70c50ab8270395a87dcafe302816e9a5ffb9b81946f7e3` |
| A/64/1.20 | `4be84cc2b57c2459b1a1e1574060d5ed6e61da68c3c93efd5d545976b356ae4e` |
| A/64/1.25 | `04d11e1ece2ccbae3428628ac8246a5479bf8a98b57c584185f1f17675e5c955` |
| A/64/1.30 | `2abbd50509e928092caa9e56f297ea309572b9b62fd60feb3d64dfafc306a306` |
| A/64/1.35 | `3a2ed1011d4f736221beded21b8dc33469fb0956a7d6fe403a3540eddbe26ce2` |
| A/64/1.40 | `313b0328085ce264cb6aa0914916e4162f6813ef5d00f2ed0d7438b09e559987` |
| A/64/1.45 | `19f17c4135f74c457fca5b0d2fcece2335c5c8671fd0f3a46f913741f11bf3dc` |
| A/64/1.50 | `2bee9a4e11c91cdd5655bab89e2c88744283a47a4da32fac7923846d0aa6dee8` |
| A/64/1.55 | `44da98c151d23f52d018912a574efab328327d08fb93ee8e78ad2d666a088f5e` |
| A/64/1.60 | `d5e946c3e3f7aac3ce6db8ab46df806bd03deaadf2548e89d58446566f0cd02b` |
| A/64/1.65 | `67e948c31937ffad76df593caf3d0595e097d74d4deb1e77a3c9bdde33cb1cd3` |
| A/64/1.70 | `bd724eaecbed37187c83164924856fcc4aedd0c1ee61b2594e615c8b6711fc94` |
| A/64/1.75 | `3bfe96a888ef2b3c997c68eaa1ba077f7003926ff841349572c51052f47f63fb` |
| A/64/1.80 | `7dd23454e2a7ba712a3f28ab18be2139ca55d4f4d42b2986f4a1b9fe1cda788f` |
| A/64/1.85 | `3d6ee3804867d68fb23e58ca00217f39a6304e04a8e4f8bf369909ceb0f18aee` |
| A/64/1.90 | `0e871f18a0cb69ce36a5e7ef3b8480bd9dfc8e19adb352f1d942aaa5a0994843` |
| A/64/1.95 | `5833e511a4adf4da2d18920f6972fa92d3892db074230f147d91fd02ed484303` |
| A/64/2.00 | `0e2c974c746db010d6a9bd8eb1352c3b251a59b6515d65a02eb8b0dcf6230a4e` |
| A/128/0.00 | `284b50b4b143ef0a069600c5f35448c514c9ab3e5864c22e5fb9d22202146ca9` |
| A/128/0.05 | `752e3492ecdbfd3456e3cd37983d8715db4bd589395d55f0ceb15387c9e3d123` |
| A/128/0.10 | `0028038faa41502d75a8acda9d625162ab6a2d0a150ce47787e435198109231c` |
| A/128/0.15 | `386a7822dbfdac8268d6df919607db2b17983d0d5b3020221b9c9252a9ba1ae6` |
| A/128/0.20 | `d54a5382f26cc5917f001892d6c3152916afb4157d5521ccfb5938c86dc2daf6` |
| A/128/0.25 | `3cf091992475da55e5459a7bf67a3dedc8b325431f64f6d347c637bc476f12e7` |
| A/128/0.30 | `4f90a940ec0db60343cbeb66591fca6cadd2afe1784525c34a6a2810de888646` |
| A/128/0.35 | `2ff26f2b914a93691db7443673dab0767a508c73931ab5ab090dd042bdc73110` |
| A/128/0.40 | `cbabeccd6e6c113595ab63cabc2afb65e52ae0622cebc35ef796a47222801cbd` |
| A/128/0.45 | `f3dbd19a86515959fbe1e1a6e00c70579aa525a4b17c1d4318385002ce04ebf2` |
| A/128/0.50 | `95c4ed26fc4593382c4bb1dfa4c36a900ebbd51566ea89b26ffa9fcf2f8dbb88` |
| A/128/0.55 | `6859f0aac6cd3550653da9fcc1f63dcb7bfecebe2c462084d0cadb1f2cfbf513` |
| A/128/0.60 | `79e81eda08e7ff3a2f598979f8f1897f98195d350fa6b870cd71f558a32401d0` |
| A/128/0.65 | `3789c2a2ec8b8bd497bfa7abd510be3dad0f098ca5c2020a03372b210f11a1f9` |
| A/128/0.70 | `ccd44168c812fa9822317cb10f31e64d909b429ca2dcc2c518494b7d365e9dab` |
| A/128/0.75 | `1f3c115b12c54c8ee8afb3f0fb0395e8ada2d74da6172c914ed7ebc22ce63be5` |
| A/128/0.80 | `d3a225d3a9485e96b6c998830d0cdd375463dfe40567c8d713bfe5a736ea0f52` |
| A/128/0.85 | `bf1ef7b17b06ecaa1248a7488cdbf12fb7e371fd29d4703f4592db645f000858` |
| A/128/0.90 | `4e9fa35dc88bf9fe883e1948264749288c41d5d91cdf309972686495d1cb7cdb` |
| A/128/0.95 | `81c2ca9852b04f83521ed647ad1c221a7033b122dec29025b13be4a32b566ef2` |
| A/128/1.00 | `1cffc4fca0a124584d0bfefb767de53fafaa69a8a0911857f7fefe937c5d9bf9` |
| A/128/1.05 | `c45029642641f3e3c162bec2882345afac9826afe5cfe0f31df8787e6653e600` |
| A/128/1.10 | `46258537f9f1e1948025a36ce64fe321e192a4ec50c6db9d6489455dd44c7d26` |
| A/128/1.15 | `47e229aa37acd5146296f6d9235292fd397323c485f5861c6b443dbb73d4cc08` |
| A/128/1.20 | `ac5fc505eaaf3e42ad0510894f738fb5af3cd20796ce512370c7776b8a553b4e` |
| A/128/1.25 | `12440d8383573861bbbfc517313a9e2b2b52f86caf8b61486a398f56a91b1f0b` |
| A/128/1.30 | `a50ae414a87062ebae63f9d6cdc0fc2cf8a18906f69e33c8d3a990372e744e9b` |
| A/128/1.35 | `f312e7b5e3731882664ad5b30dabac389fcf41973f00e8f6e59ec5c09f9e476f` |
| A/128/1.40 | `022ec5240970b5c35d62d7264a66c51e16e3e5c5369e831c972a2e49087e3206` |
| A/128/1.45 | `df7f268b4380255bc1cb227ba78341ad4ff21e244934248dcb6c7d0d663d7078` |
| A/128/1.50 | `fd3d6198ab1539544aa7d83c0a426afbee5d150cbeb30cb45723ad9c1f075e1a` |
| A/128/1.55 | `bbf1e548e9bb7dd7c933ccdcea27d30cad54dd4b6b9656957d4164dbec241a1c` |
| A/128/1.60 | `23ee812737f32cb5a836b55dcdd03f0bf901c1044f2361f4210abe411d5d026f` |
| A/128/1.65 | `8e2f252895f7c0bf1e6557e85c94210278d1e0d8a61779f547cacf0896d6e8de` |
| A/128/1.70 | `b2e446c19983918828ec1386669cdc6cf31e4bbe32fe55c3b649954cc0a16c18` |
| A/128/1.75 | `9dd3de23e7346fd71c247fac5ae2de3483c9474530f4d94f94b00f29fbb7785a` |
| A/128/1.80 | `72122ea96c4f287099a3efafa00e8c5f8ad6fda21f90b8dd27968c55ac6b4439` |
| A/128/1.85 | `a12c41668136ff97029890e705b377d8d2f33b7fc550e0809732f330f1514739` |
| A/128/1.90 | `da37b14df4c0c11427eff9983a17c2e61e9d3fdcc6ee8daa47f6aa476b009144` |
| A/128/1.95 | `6b2c93ebf303a56745fdf5ceab7f9ba81d8d6e0a79b248ff8df11bf42f3f5fc5` |
| A/128/2.00 | `8eb9f2f0ec45794108ce11ef9bf3bbc258a1f30934489c2368a680762e85210c` |
| B/16/0.00 | `56aab168fc596acc3a2c1bcf62c638756d5e5fb45c7de7bd947c8c07f0905ad1` |
| B/16/0.05 | `0e80654049eaccc80f8f3d98f1b64992545a00bb2c8e3a037b64c75c4d590400` |
| B/16/0.10 | `90d1b0be0558f897eb1b6c0f8d6a9d8ada626d7ceef8203004e6166091268ff2` |
| B/16/0.15 | `85114541f3315297092bd6c92df695c9ff6873afd321ad1dc1b95cefce2fcf69` |
| B/16/0.20 | `44a8c2868f02fad9046e18fe8cc180216b8315b7c542940cacbe96843d9832ba` |
| B/16/0.25 | `9000f9683c94e2481ee78e670fff85551edb80ab63444b2b94e9f10c5670fbfe` |
| B/16/0.30 | `daf0636e746c9b3aef98168bcd933801bf3956227b7a22f23e75f59046644de1` |
| B/16/0.35 | `b815ac1c682313743c3b0f711401e6919c353178b4c09b0307b116390ef0d6fb` |
| B/16/0.40 | `002733417639f854cb6649581d7b4e082e3cf4c6f63cbb3dcfe077a46546cd8c` |
| B/16/0.45 | `ad4d9c2a17b71cd8ebc4ccb6e5ae4147d96d60e3b5a9e1f8a9b8505cf02ff1d9` |
| B/16/0.50 | `1ad540da700540457d14a06e235d27c6cc2ba835dc2af6387113f7077691a754` |
| B/16/0.55 | `df78afb7d7cd71300743bc8aa06cd39cc664c3c2cf766094392ac662de4842ac` |
| B/16/0.60 | `45bf5a83917d5fddcef2312a080e234567ee9eaaebdfaf2fa67046ed9326263d` |
| B/16/0.65 | `cd8b6e00b2886a6f8a2f36ff1b365aa86503e8cf391e402eee9b1dd94a2777b0` |
| B/16/0.70 | `e0cf7b68f5ee6b5f32c8aa88349a1ad934e1f9ea014207c778698123f16217ee` |
| B/16/0.75 | `7679a9d2908971ded195d6715499081348ce88831d70ef375cf73e25f16a63b9` |
| B/16/0.80 | `2d6ea15dd14a2c3335fd9226493971d4ecab5800b9e7d0821b95062961757a2b` |
| B/16/0.85 | `53d8d6bdf4d163a6d67aaa9dce3353358a4f6188fceb6851257e8d1a63f56ad0` |
| B/16/0.90 | `1b1cab948be2b34736860d27748a4578608d6e6a30c531308407750229f4f8c5` |
| B/16/0.95 | `72a40fb128f81547a5dd5fe5509d46f89e994e2485fcf94c961d02e94ae1eb82` |
| B/16/1.00 | `e4e5926fb97468b2f174c3aea0eb830e8e211b74d68dd3f7d36a90e094ada8fb` |
| B/16/1.05 | `203b115da1d15a9e0ce697f43648945943559e11b02cea966862bb486f491e45` |
| B/16/1.10 | `a4946b6cf3832cb1eaef20068e07cd0a6c91f6541eb917203d60234dd781c451` |
| B/16/1.15 | `4d0d23a42aece08080a24d0a552f862647235079bbd3edc2bccad34b91957b35` |
| B/16/1.20 | `a15e8c3d476e8760327ea97848d9ee228287dfd0a99711c33eb6010733ce0cfd` |
| B/16/1.25 | `e48053ab0dc1a900028b210d07b769d93c658ac9bcd44e86d03e367a3c506939` |
| B/16/1.30 | `71dc090ed68d80d9dd4bfc9aad516e99d64554bb600deaeb019fbf514ab0b1b5` |
| B/16/1.35 | `386bb215c02538724bb989031c7c5262f3d6385d031a783f0f37192db0694e78` |
| B/16/1.40 | `71adb09c0016d3f1ce41048b28c729aa664cd142a44583c815784d8ef4f4b2b9` |
| B/16/1.45 | `f01e160f4e5866c70a562035849481fc484d6a82e17059b698fd7a394ad3ad58` |
| B/16/1.50 | `14635bb52064f768d534f4433fef0a5876cc3cd0cc47c9ccb3c60f53529f3a05` |
| B/16/1.55 | `cd6716cbe0f37950bea14fd35ce94fdaaa344fa51b39e43aa3e1a8f74322bb5f` |
| B/16/1.60 | `529c572cace12706d92416b444c32dca2050a16556f6a5cc307e93b2aa15dfc2` |
| B/16/1.65 | `19377330a3f4a5a1f646b1ff91502fe62785bf04d81e602f98abd8bdf036d42a` |
| B/16/1.70 | `2248daface3a9f4b833f0455073965b69f19f6fe143736a8ff228e5e2dfa6aeb` |
| B/16/1.75 | `92a4c0981e2480329b89dbbff7663aef2c5e6c5f78ce59b96a189ca6bf8174ff` |
| B/16/1.80 | `c34c75ab17326d1510b71d92ea9737dc1f261debb840e40bd81fd715dbede37e` |
| B/16/1.85 | `2095401c9ff62863c269d7f401f64a6b22fa32895b12ebf3d9cf24050d0349dd` |
| B/16/1.90 | `6cf81b660c0b9c05db9dbdb795a8ec13fbe3d2c2a402ba9dc745ce32c8c27b4c` |
| B/16/1.95 | `9c4cc8767703a2dab209a5f538d85e2fabb7017074f39857af2e696aec9ef94b` |
| B/16/2.00 | `819c0cac9c80db159fd238f319718717e09a29e7f524607a949d77d0a61408dd` |
| B/32/0.00 | `80b3668136abb0b4ab1fa1e2cda3aba4797485fc425acb490a13b22b9224f9af` |
| B/32/0.05 | `d417e224ab9426d50892327713b6c0f52712465cf047d62a0058804fda9df8b9` |
| B/32/0.10 | `bbbfb1cf217794d18d5e2b2ca60f5ef9aa8f10d8bde0e6eb145303add40e86f3` |
| B/32/0.15 | `08d24f413127fa8af5f5ed2c23b340c668e56723e7bcee4401c1e7993338a0c0` |
| B/32/0.20 | `0ab69627ecde46d840cff8feea2f115fa5874ef5df8d7e47148faac6d77e5c54` |
| B/32/0.25 | `77fb9e6d4d04c7ed3e73344f223e892b6f443fb739742200dc7ed3f30a48659a` |
| B/32/0.30 | `e469d8d8e4549962cc2219cd6bbf02c3b0897994aaae3ad9eb1f880c4c37ec3e` |
| B/32/0.35 | `8ad63f7726f1a603465653171e37cfb38aea662c99b09f8335c153600a7a55c6` |
| B/32/0.40 | `88be61432835b190413320cdbfd867c176a8715369e22393e2158646b9345891` |
| B/32/0.45 | `6b258f2840b8bdd464881a518d479f870fa323c5ed77b090c674d8ca7c92d818` |
| B/32/0.50 | `c5bb87bf5960a26872852cb565e032ac92a451ff3f541c501c1339038f8294bb` |
| B/32/0.55 | `69e762a2c3a2622c0b693ac1ec475f522a8dff222ab8c035c4a55a5e08e07cef` |
| B/32/0.60 | `72d799721cf2e38f6eceb7a86c2d5a17eef6d8b498cbd7a1c922813a23059a0c` |
| B/32/0.65 | `53c1c49d698cf49d0893820de6eb577b3f44428188700780d0e0f4192e57e1a0` |
| B/32/0.70 | `5b21ec19657a1cd4744cb86cbca791958ad75e097afd7050ef83123da3d247bb` |
| B/32/0.75 | `38e8a1e293229426151efefbf697d0c2ebc27742535a712c21c0f2bed07cc4f4` |
| B/32/0.80 | `ff0e97cef42a262871352a688a45748ce2b585b07585504b6a8e212c38e44944` |
| B/32/0.85 | `f5c0934d601fb69f6e48b061f4bbdba0aaa3d8fa6624e2139b053763721fffa0` |
| B/32/0.90 | `5862d148b0d5469493ab6fc1374a70b9414932449d3477cbf25e7e2032e9ba64` |
| B/32/0.95 | `e165ed382b0a83766b4b9dd09d1742fdb5a50fb099ae946f0bc0f4920985ac63` |
| B/32/1.00 | `392eea49c14961f103a9f52e877eef14e034a337ed61eada5d1a911001634b85` |
| B/32/1.05 | `6b3de6844798e424eb85ebe831604605191807b12291920e4feba51a82760609` |
| B/32/1.10 | `15d17f8c858a37b97ef65d125e76fe34803dde2c2a627dd838a848eb1d31fee5` |
| B/32/1.15 | `92d4d2093810fffcdff7ad8d90bd9e59054dae48d15ff97330176fc074e68701` |
| B/32/1.20 | `d5fe11e308e2d30ec53bc3c7ccc30c6f7815370c9dd937e0ef57e725fc422f71` |
| B/32/1.25 | `3a6f3bf4b053d45989d45867fbef3eb20f9ae3d104809ab3ad9740fdeff971eb` |
| B/32/1.30 | `d3b8b6cdb303b67ab82df47374aa70fe13673fb8f535b579605aed31e0bf769d` |
| B/32/1.35 | `e93e5ff82e9147e1de4345f2f566ad216f477a9719d8b19f9c1adfa2bdff890b` |
| B/32/1.40 | `0181a188f05af94631deb81788c75c64d1fe60fb2c9a3a82d417b6ba8980a61e` |
| B/32/1.45 | `280cc20c680c2c9a535d1f3419a6bf8027e1435197677cec339a433d9d653bb6` |
| B/32/1.50 | `ee6ab3ed29aedece81afe4d30f14bd93c6ded39d0ebb9b41dc6728594f8b136e` |
| B/32/1.55 | `1eb2103dbe339995ff1f077c7b49ec2349c081fcc1ef828fef830d785a4ad053` |
| B/32/1.60 | `fb4e3a9f0bacfd249bd5f3554d4d98da25b40b8f7cdd43da02653b2a68dae09f` |
| B/32/1.65 | `d76c774ed7cefca2d1af2e509f28c2b177a18ab5ec13e14de1a3bb4f57c01db5` |
| B/32/1.70 | `c2b13140b2a05d323b48515235445bc814bb415f13be120c3b1bccbebaaadc96` |
| B/32/1.75 | `a6186da1e04a54dff4e5e2f4f1b1b822895170a8e72479752260fa40ece59041` |
| B/32/1.80 | `61e243d0ef7e7d7fe272b809eab36d4806fbae940661dc473adec68aa997a6f2` |
| B/32/1.85 | `42f26779ae4e78695b083d14c9034ff681d585673906d1a3aa1f2c5b664dc04b` |
| B/32/1.90 | `183de210358eff539cd55cb3a848123bd9bff668b74627f3e7c18591845a9b57` |
| B/32/1.95 | `76b58af4fe84107efc2155849d60128fcbacf35350c0a6df79f47a4c25f278a4` |
| B/32/2.00 | `4d282ed9b5aaac8b76df2b45fd1e0f548e36f24391770faacbed2f0dfe711fbe` |
| B/64/0.00 | `9b94f1a5d653992a6d515591e7342897f9e20b7641671db05d8bbbbfedeb83cc` |
| B/64/0.05 | `ab2fab3a465cd4da87ce0b6b7ca652317bf7a71eae21535035f403e5d7112f80` |
| B/64/0.10 | `aafe6aeb0f99d2b2a52bbd96a6d9dcf6dad3b28f4cedf799057d9872391ddb62` |
| B/64/0.15 | `5e573edc17526a9399655a662c364fcb988108151037bd23e93a00e2d8fe9eeb` |
| B/64/0.20 | `121e769d34de6d204e81456f1c83574e6211f23129e3e0eb10ba324355cc675b` |
| B/64/0.25 | `92d1a62dc77a7bebf3460c8849897172799e10da7e88d525e5c7c9756764c76d` |
| B/64/0.30 | `c90213950a15907cd312c2d45229667a212370cef506d3db90b781a7e25d1969` |
| B/64/0.35 | `a9b6e471fc82550b2011e4d2128f59fd77aa87e04f6fdc8744b2f6ddb78578fb` |
| B/64/0.40 | `79d3c8b6636be9b0bd3310a47daccfafb87422eb74d9e8aaedcea01f22f52039` |
| B/64/0.45 | `4abb565010df662652bf2591b05afed4e2ec91c1b37c61c616cf5543e3ea2ef2` |
| B/64/0.50 | `05d3b1a307f5a84ef88b31278b230ff1ab165d2fa8a6a2563415895d6d5ce4ea` |
| B/64/0.55 | `c186e35824ea377d9f2e5e37691b58ecafbdda96f8b6ea4bba4f3a73aac1b579` |
| B/64/0.60 | `6d9415d0c565d7b5d254902976be7cad61de689df1a06d8f16fd4e6c9178a197` |
| B/64/0.65 | `5ac57e8c49e947b1d18e322bdeeee8fa3b5c5f24285b7582a1808c8fb972f8a4` |
| B/64/0.70 | `9b4acdbd2753a6f2fb0b5d7ab28118d59619cdb8168b944792fc4b4969cd48a6` |
| B/64/0.75 | `3dc48c4ff4718015f4d88464d0d22f780c5cca56a8931fbdb636d92d99db6b1e` |
| B/64/0.80 | `b7acb6b22e967980d10245a9f2ca3e714f7ca74971dd7948bade6f9a148f5cd1` |
| B/64/0.85 | `02ecc2fafd05b969945a580dc98a95f067e1221f12d1145b00dc3ccc5f195c2a` |
| B/64/0.90 | `d8b57135f812233d7d5b97a2104f46a4a87f79b4e6716b6cf7944501919e16a8` |
| B/64/0.95 | `14b1d05d7dc8b46278b0aba93a313da39a3d6fee7300483e7c1b26391a05c37e` |
| B/64/1.00 | `3aff04a12eb8b5b4f490daa436777b994f2287b0fd44943962ccf69fb7e3875d` |
| B/64/1.05 | `b72658ac4bb3653ff6403f3746b9d1be648f1d7af26789f849c40af5d1fbcb94` |
| B/64/1.10 | `2107a210712ed9c4e7b500a88b31b7158a1188bef8ee61199bc48a4ad091cd3c` |
| B/64/1.15 | `1611673c632dddb45711d64b39119f29cd1e1c6414af8d05b6c245df07a466cc` |
| B/64/1.20 | `3e37a8f4b8e996a6fc1098c650602091d0f8001bb35350c28d784678e6050c70` |
| B/64/1.25 | `a0941590a7a3dbed08107e90f29b5b6a7de550f109d7d770bfcb8ac46344ab28` |
| B/64/1.30 | `901dd44e2bedecff1161690437456cc5a1122dcc5d7e536ca09197a7edfaebd2` |
| B/64/1.35 | `6860c71d4ab0019a824ad079435ca46668344972a056ad5184c4becca0b318fb` |
| B/64/1.40 | `f78007877bbac056275dd9b9324733c1626e914528d529eb5d6086e1788f635f` |
| B/64/1.45 | `079e676698de418d17194749433b231b909f4ede52dfe18dd6c8a91d30179a6e` |
| B/64/1.50 | `01f9e271b7acde03d20912622b4af70e14bf7f9e3505f5bbe27950f08b125962` |
| B/64/1.55 | `79539c6db69e4e901b729f59d23c5dab891608b3cf0755ec3018959b9cd128ee` |
| B/64/1.60 | `2dd53699ed94a3d0b3de552b76891f770bb323861745b9c413751e090e2ac5d8` |
| B/64/1.65 | `969422a18b051dd3de7f037a9b074692eaca5d17644f7a10adb7fbc9fbb0f99e` |
| B/64/1.70 | `ed03122c8c748d25924101d0d0498d62711c20fed2af92e344aa83ba2b8d436e` |
| B/64/1.75 | `de841675ee29d24a5e8eab34cc21b3fc7ac248957dc2f4adc96dfd99144c99f4` |
| B/64/1.80 | `1db35b80b046509555f9273a5c6827bca21bf36404606282aadf8744efea3548` |
| B/64/1.85 | `7aa4b4b51a83f2ef84cfd7303b34fa62fbe154487c1a81755b2f72f5a785ac70` |
| B/64/1.90 | `38f8eaf707136b3f3cfb6f1a18308856fee39f0c8db0040d4c6fc33db1851edc` |
| B/64/1.95 | `f69f964bf7b7f5240da72f9d344f6e6f4194fb600658c21dadcdc3f8d549e273` |
| B/64/2.00 | `6f4a3e3547f3cdf7d148e7584b4cc7b8b9da549c93b13663f13cbf822999d5ac` |
| B/128/0.00 | `84528a31bab163ce7cc477e28e0276a5d97551cdbeb5243cefb497084fa7750c` |
| B/128/0.05 | `959272c9e345992b921df3ac7bb8b18a98445d8ca556413f7e131cdfcadadd94` |
| B/128/0.10 | `36613d1b8554b8f66e0327e701ee65378d63476aad275fb1cdc19cbb0516e903` |
| B/128/0.15 | `f09b4815e8cd669f3a1711caa78fbba64e80d9124aceac5e0f50ddd97f195f3b` |
| B/128/0.20 | `54e5e9bba4d8535a6bab5ff746df3b1552565ff7409678444ab26da727a28de8` |
| B/128/0.25 | `e7042191f07031fa7be8baa8ab5fc834e72ff5142bdf05a3b73589bc9de43a0e` |
| B/128/0.30 | `4da3278a3be9ed32a68475c73dac4c1ffd3e6fc6e5c8954b73bbacb68e7bec0c` |
| B/128/0.35 | `3530c1097e57c86d8b874bc7ce9b3f7f9aae2373a3598665697c78ae54f55c0f` |
| B/128/0.40 | `1b7f39bfe81e5d00ffa470910d98f50f9add2dfcd0bc7410c1c011e18ba12491` |
| B/128/0.45 | `f193bff2066867ef6e91b7dc0c7b55b497bd1be6b27edf2ff20d95644d023105` |
| B/128/0.50 | `c973e6403ca2848bc655195513c5563b4b98aa52dbfaeb5c689b04ed57e4d27c` |
| B/128/0.55 | `cfd3ba2408f899802166b80130644b6791de61db7d5ff2783fe4be3a458b5b97` |
| B/128/0.60 | `c26e4a44550c55d7c11aecd6309071950f1de182eb4b72aa9250790fc75eddb5` |
| B/128/0.65 | `e8fd24c2fea5dbf07b7cdfc031de670c498b647b6c86c3e8823cd565b36bd701` |
| B/128/0.70 | `e853951c4268e0e72430d73b380994bb4b7961e1221007e98a37f700aa7d57e9` |
| B/128/0.75 | `95fe06e97e89bcae814bca12eba7ec8a99daa335a344e2c3e12ede20f8260b53` |
| B/128/0.80 | `6e129617be1307670587210f276e8d96c234dd23d47ead0533cc516919db1fd3` |
| B/128/0.85 | `f8abfe8687efae929495b8489a48a71e3e5ed792a1e71f230d331015776da246` |
| B/128/0.90 | `880cc3b81868cc059d2f65e932a9f1584f0d132aa74be2594d4adba81b9655a0` |
| B/128/0.95 | `05761b01f0ad90130bc15bbe15465db8d8757c845fa71de86e2bab9a8d280e57` |
| B/128/1.00 | `d59f8660c881f443832e555b7a3183f85dfd2f537f87d437646316317ab64e08` |
| B/128/1.05 | `8685fc2a3582b1e1ed74bd82f4c800faa9b1dd1a8bac0a1289b4274006f7bb73` |
| B/128/1.10 | `75cc57eabec99b0930a6dcfd8bee402146bbdf75581042fa093598850d76ac2b` |
| B/128/1.15 | `042563e8520f79fc9d75d1e651842051d5c17e6524bd99ccf90304badd94d8fb` |
| B/128/1.20 | `dc3367f10154c0875a92f7f33ec0e68e08d4e8bdddd64fb73e93a8a9ad292850` |
| B/128/1.25 | `33a0ee80b554377db043b34e15f7a8e94ed9e34774cdb884e8c4585912a93a38` |
| B/128/1.30 | `cf4769cab45764e1f6074e2286cc450c15fcefad11e8c20e1e948be49df70738` |
| B/128/1.35 | `ce30b325eca765dd2767df05fb60b614a49a608fbbe10065b7933645346a2501` |
| B/128/1.40 | `85e9cec1965998bb5c63b25d24674a574358e25f2c98f54c4e22a515bc9880ff` |
| B/128/1.45 | `26e7f263fb6024d50eae2de2287b341fc83188a5fbf79a4d33f1418b9bee24a4` |
| B/128/1.50 | `c4793d7619888a2fcb41536dfe007ae46b4ed56d40e73419e0b1be83b2380589` |
| B/128/1.55 | `fe3f8d30ec5189e2c89696e955b26c5b2e5e719ae5ded9d42d5e4ae830990129` |
| B/128/1.60 | `84e05c14d5f05d9c3c1aac9ea7ead34df0b5eb407b7a2e0bee28cc519a4e5765` |
| B/128/1.65 | `93d77759890f6f105f3ee369aab80e21766b15d485a3f8715f5ff5ca72175ccd` |
| B/128/1.70 | `226d378e85fd666b35f93fd2e4f86d882bc5daf329f62620dcf0e70306c5d7e3` |
| B/128/1.75 | `10ccf78b1c833c1890ac9b3fdbed11774754e5dbddfcb03a6cf993e38b5d8a42` |
| B/128/1.80 | `eb96edc645eef24625131dbb57753c27567f4e7cd8f2385a81efb7a24983680c` |
| B/128/1.85 | `ff61298ac4a9c38288848d9ec7e2d10712b022df1bb8ea6aa45e94a6b01ea560` |
| B/128/1.90 | `b43a06322990c20dd2f57e79b5be143fef379257031ee8c4439a588d2bfb7063` |
| B/128/1.95 | `8beea0b6936ead6da09e0c207be64a0dd57b3024205c0a64122877620a9071a3` |
| B/128/2.00 | `c46635e059f89c8ee6312d1c06967e9469de505e549441694646412439fb5f18` |

## Grid

- models: pure, soliton, gru, gru_pm
- T: [16, 32, 64, 128]
- sigma: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
- confirmatory seeds: [0, 1, 2, 3, 4]
- runs: 160

## Seeding

- Every stream is seeded from a named path by phs.tasks.derive_seed (first 8 bytes of SHA-256 of the path, little-endian).
- Training data: ('train', task, T, seed), numpy PCG64; the same stream for every model.
- Validation: ('val', task, T, seed), 1024 sequences; its noise ('valnoise', task, T, seed).
- Test cells: ('test', task, T, sigma), 4096 sequences; their noise ('testnoise', task, T, sigma, seed).
- Model initialisation ('init', kind, task, T, seed) and training noise ('noise', kind, task, T, seed), torch CPU generators.
- Confirmatory seeds 0-4; calibration used seed 100 and the namespace 'calib', never 'test'.

## Metrics

- **Acc(T, sigma).** share of test sequences whose last-step prediction is right (task A: the marker; task B: the final state), per run and cell.
- **sigma_crit(T).** max{sigma_g in the evaluation grid : Acc(sigma') >= 0.8 for every grid sigma' <= sigma_g}; -0.05 if Acc(0) < 0.8. Per seed, mean over T in {16, 32, 64, 128}: mean sigma_crit.
- **lambda.** per seed, the least-squares slope of Acc(T, sigma = 0.10) on log2 T over the four T (accuracy per doubling of T).
- **B mean.** per seed, mean over T of Acc_B(T, sigma = 0).
- **R_slip.** per sequence, the number of (token, channel) pairs with |Q_t - Q_{t-1}| >= 1 on the holding segment (task A: t = 2..T-2; task B: every t), Q_{-1} = 0; mean over the cell.
- **erank(Z_T).** exp(-sum p ln p), p = s / sum s, singular values of [cos Theta_T, sin Theta_T] in R^{N x 2d}; mean over the cell.
- **C_step.** forward wall time per token for 64 sequences after a warm-up, median of 3, under the confirmatory parallel load; plus analytic FLOPs per token.

## Verdict rule

The goal's reconstructed rule, tightened by decision D11 (room) and computed by the frozen analyze.py. Comparisons are one-sided exact permutation tests of the difference of means, 5 against 5 (252 splits), alpha = 0.05, Holm within each pair (A1, A2) and (G1, G2); a flag holds only with significance and the minimum effect.

Parameters: `{"B_margin": 0.05, "V_acc": 0.95, "V_min_seeds": 4, "acc_threshold": 0.8, "alpha": 0.05, "lambda_floor": {"T": 16, "acc": 0.8, "min_seeds": 4}, "min_effect": {"lambda": 0.02, "sigma_crit_mean": 0.05}, "room": {"pure_lambda_max": -0.02, "pure_sigma_crit_max": 1.95}, "sigma_crit_below_grid": -0.05, "sigma_ref": 0.1, "verdict_models": ["pure", "soliton", "gru"]}`

Flags:

- `A1`: Soliton > Pure on mean sigma_crit, task A (Holm-significant and delta >= 0.05).
- `A2`: Soliton > Pure on lambda, task A (Holm-significant and delta >= 0.02 per doubling), and the lambda floor.
- `G1`: Soliton > Control-GRU on mean sigma_crit, task A.
- `G2`: Soliton > Control-GRU on lambda, task A, and the lambda floor.
- `lambda floor`: Soliton-PHS reaches Acc >= 0.8 on task A at T = 16 and sigma = sigma_ref = 0.10 in at least 4 of 5 seeds; without it a flat slope may be a flat failure (decision D19).
- `B`: Soliton not worse than Pure on task B: 5-seed mean of B mean lower by at most 0.05 (point estimate).
- `V`: results/gate.json at the tag shows every test passing and every mutant killed, and each verdict model reaches Acc >= 0.95 on task A at T = 16, sigma = 0 in at least 4 of 5 seeds.
- `room`: Pure-PHS 5-seed mean of mean sigma_crit <= 1.95, or Pure-PHS 5-seed mean lambda <= -0.02 (room on one metric suffices, decision D11).
- `complete`: every run of the three verdict models has an OK manifest.
- `admission`: analyze.py refuses to compute unless every manifest is one expected run, once, under its own file name, from one tag commit shared with the gate, with the registered configuration, python/torch/numpy versions, test-cell namespace, digests and sizes (decision D20).

Verdict, first match wins:

1. `INCONCLUSIVE` — a run of a verdict model is missing or FAILED
2. `INCONCLUSIVE` — V does not hold
3. `CONFIRMED` — A1 and A2 and G1 and G2 and B
4. `CONFIRMED_VS_PURE_ONLY` — A1 and A2 and B, not both G1 and G2
5. `TRADEOFF` — A1 and A2, not B
6. `REFUTED` — not both A1 and A2, both point differences Soliton - Pure below the minimum effects, and room
7. `INCONCLUSIVE` — both differences below the minimum effects without room (ceiling)
8. `INCONCLUSIVE` — otherwise; the remedy is a new PREREG version with more seeds, never seeds added silently

## Secondary measures (not in the verdict)

- Spearman rho(R_slip, Acc) of Soliton-PHS over the task-A cells (T, sigma), cell means over seeds (prediction: rho < 0).
- erank(Z_T) of both PHS models per cell.
- C_step (ms/token) and analytic FLOPs per token for every model.
- Trained kappa, g_lat, g_pin, mass range, ||J||_2 (medians), the calibrated gamma, and the LATTICE_UNUSED rate.
- sigma_crit restricted to the PDF grid (min(sigma_crit, 0.35)) next to the evaluation-grid value.
- Soliton-PHS against Control-GRU-pm on mean sigma_crit and lambda (exact one-sided p), not in the verdict.
- The starter's estimate Delta E_PN ~ 2 g_lat is not tested; no barrier is measured (transcription note 8).

## Decisions

- **D1.** Pure-PHS, Soliton-PHS and Control-GRU have the same state dimension D_state = 2Nd (GRU hidden 2Nd = 128 for every (N, d) of the goal). Control-GRU-pm has the hidden size whose parameter count is closest to Soliton-PHS (per task, because the input port depends on the vocabulary); it is reported and never enters the verdict. Every model's parameter count is reported. *Why:* GOAL decision 1.
- **D2.** The gyroscopic term is the exact Cayley rotation G(h/2): P <- M^1/2 cay((h/2)K) M^-1/2 P, K = M^-1/2 J M^-1/2, by torch.linalg.solve, in the sub-step B -> G -> A -> O -> A -> G -> B. The PDF's explicit kick exists only as gyro="explicit", a negative control of tests (1) and (2); the runners never set it. beta = 1. *Why:* GOAL decision 2; starter notes 4 (C4-C5): the explicit kick has norm sqrt(1 + (h lambda/2)^2) > 1 and diverges to 1e58; test (1) at a fixed horizon kills it (orders 1.65 and 1.36 against 1.91 and 1.96).
- **D3.** The five kernel tests of GOAL decision 3 (tests/test_core.py) plus task-generator and verdict-rule tests (tests/test_tasks.py, tests/test_analyze.py); every test is burned by a named mutant in mutants.json, and tools/run_mutants.py requires each named test to fail on an assertion. *Why:* GOAL decision 3; primitives pattern 'A check counts only when a mutant breaks it'.
- **D4.** Noise: in PHS sigma = sqrt(tau0) enters through the O-step only, c2 = sqrt(tau0 (1 - c1^2)), P <- c1 P + c2 M^1/2 Xi, every sub-step; in GRU h_t <- GRU(h_{t-1}, x_t) + sigma xi_t once per token. Training draws sigma per sequence uniformly from the PDF grid {0, 0.05, ..., 0.35}; validation draws the same way from its own seed. The two noise channels are not commensurable: Soliton vs Pure is the clean comparison (shared channel); Soliton vs GRU carries this caveat. *Why:* GOAL decision 4.
- **D5.** Task A: input tokens 0 (blank), 1..7 (symbols), 8 (query), so the input vocabulary has 9 tokens and the readout 7 classes; 'V = 8' of the goal counts blank and symbols. Marker y at t = 0, 1; t = 2..T-2 blanks with distractors at p_dist = 0.25 from the same alphabet; query at t = T-1; cross-entropy on the last step only; chance 1/7. Task B: q = 5, commands 0..3 uniform, s_{-1} = 0, dense cross-entropy, Acc(T) on the final state. Test cells: 4096 sequences per (task, T, sigma) from their own seeds, pinned by SHA-256 below. *Why:* GOAL decision 5; the transcription leaves the query token outside V (note 10).
- **D6.** One AdamW (weight decay 0.01), one gradient-norm clip (1.0), one batch (64) and one step count for every model; the same data stream per (task, T, seed) for every model (noise streams differ in shape and are per model). A separate model per (task, T, seed). The checkpoint is the validation maximum of accuracy (ties: lower validation loss), evaluated every 100 steps on 1024 sequences with their own seed. lr and the physical settings were calibrated with the same budget of at most 12 configurations per model (S3). *Why:* GOAL decision 6.
- **D7.** Theta_0 = P_0 = 0, so erank(Z_0) = 1. kappa, g_lat, g_pin by softplus, M by exp, J_PP = W - W^T, all learned; initial kappa 1, g_lat 1, g_pin 0.5, M = 1, W ~ N(0, 0.1^2). LATTICE_UNUSED if a trained Soliton-PHS has g_lat or g_pin below 1% of its initial value. *Why:* GOAL decision 7.
- **D8.** gamma is a fixed, calibrated setting, not a learned parameter. The secondary measures report the calibrated gamma, not a 'learned gamma'. *Why:* Counterexample to learning it: c2 = sqrt(tau0 (1 - exp(-2 gamma h))) -> 0 as gamma -> 0, so a learned gamma can switch the thermostat off and sigma would stop being a noise level for PHS. The PDF's Algorithm 1 takes gamma as an input and its BPTT parameter list is {kappa, g_lat, g_pin, M, J_PP}; only GOAL's list of secondary measures said 'learned gamma'.
- **D9.** Validity V covers the three verdict models (Pure-PHS, Soliton-PHS, Control-GRU), not Control-GRU-pm. *Why:* GRU-pm is auxiliary and outside the verdict (GOAL decision 1).
- **D10.** A missing or FAILED run of a verdict model makes the verdict INCONCLUSIVE. A FAILED run may be rerun only with the same seed and a recorded reason (tools/confirm.py rerun); its first manifest is kept under results/failures/. *Why:* Training is bitwise deterministic per seed, so a rerun reproduces a numerical failure; a rerun only repairs an infrastructure failure. Scoring failures as zeros or chance would bias lambda (a flat failure looks like no decay).
- **D11.** Tightening of the reconstructed rule: REFUTED also needs room on at least one metric: Pure-PHS 5-seed mean of mean sigma_crit at least one grid step below the top of the evaluation grid (<= 1.95), or Pure-PHS mean lambda <= -0.02. Without room on either the verdict is INCONCLUSIVE (ceiling). *Why:* If Pure-PHS holds the whole grid at every T and does not decay with T, no Soliton advantage is measurable and REFUTED would report a ceiling as a refutation. Room on one metric suffices because the hypothesis is a conjunction: a testable conjunct without the effect refutes it. The goal allows tightening before prereg-v1.
- **D12.** R_slip: per sequence, the number of (token, channel) pairs with |Q_t - Q_{t-1}| >= 1 on the holding segment t = 2..T-2 of task A (every t on task B), Q_{-1} = 0; averaged over the cell. erank uses p_r = s_r / sum s of Z_T = [cos Theta_T, sin Theta_T] in R^{N x 2d}, averaged over the cell. C_step: forward wall time per token for 64 sequences after one warm-up, median of 3, measured under the confirmatory parallel load (a relative measure). *Why:* The PDF names the measures without fixing segment, normalisation or batch.
- **D13.** B is a point estimate: the 5-seed mean of mean_T Acc_B(T, sigma = 0) of Soliton-PHS minus that of Pure-PHS must be >= -0.05; no test. *Why:* As the reconstructed rule reads.
- **D14.** The evaluation grid is sigma in {0, 0.05, ..., 2.00} (41 points); sigma_crit, the room rule and every test cell use it. Training and validation keep the PDF grid {0, ..., 0.35} (decision 4). sigma_crit on the PDF grid alone is reported as a secondary measure. *Why:* Counterexample to the 8-point test grid of GOAL decision 4 (calibration/probes.jsonl, seed 100, calib cells): at T = 32 after 12000 steps every selected model holds Acc >= 0.87 at sigma = 0.35 (Pure 1.00, Soliton 0.99, GRU 0.98, GRU-pm 0.87), so on the PDF grid sigma_crit(32) = 0.35 for all four and the comparison is censored. On the wider grid the same probes break at 0.85 (Pure), 0.50 (Soliton), 0.50 (GRU), 0.35 (GRU-pm); at T = 16 Pure breaks at 1.20 and Soliton at 0.75, and Pure is at 0.28 at sigma = 2.0, so 2.0 leaves margin. Minimum effect stays one grid step (0.05).
- **D15.** 24000 optimisation steps for every model and run. *Why:* Steps is a shared calibration setting (GOAL S3). With a constant lr, one seed and one data stream, the first k steps of a longer run are the k-step run, so the validation-selected checkpoint can only improve with more steps. Probes: at T = 128 after 4000 steps all four models are at chance on both tasks; at T = 32 task B rises from chance to 0.47 (Pure) and from 0.51 to 0.80 (Soliton) between 4000 and 12000 steps. 24000 steps fit the time budget (estimate below, limit 48 h).
- **D16.** Calibration (S3) used seed 100 and the 'calib' cells only; at most 12 configurations per model, every job logged in calibration/trials.jsonl (88 jobs in 5 rounds). Objective: mean over tasks A and B of the best validation accuracy at T = 32; within 0.01 of the best, the lower mean validation loss wins (rule fixed after round 1, before round 2). Selected: Pure p07, Soliton s07, GRU g06 (10 configurations used), GRU-pm q07. Pure p07 and Soliton s07 have identical settings apart from the lattice. *Why:* GOAL decision 6 and S3.
- **D17.** Probes (calibration/probes.jsonl) retrained already-tried configurations with their seed to measure them on more sigma values, other T and longer training; they spent no configuration budget and selected nothing. They decided D14 (grid) and D15 (steps) only. *Why:* Measuring where a selected model breaks is not a new configuration; the probes are logged like trials.
- **D18.** A non-finite loss or gradient stops training; the run keeps its best validation checkpoint (diverged_at_step in the manifest) and is FAILED only without one. *Why:* Calibration showed divergence at lr 2e-2 and with S = 2 at (32, 2); a late divergence after a good checkpoint is an optimiser event, not a missing result.
- **D19.** Lambda floor: A2 and G2 hold only if Soliton-PHS reaches Acc >= 0.8 on task A at T = 16 and sigma_ref = 0.10 in at least 4 of 5 seeds. *Why:* Review r1 (Codex, P2) counterexample: Soliton at chance at sigma_ref for every T has slope 0 and would beat a Pure-PHS that decays from 0.99 at T = 16, giving CONFIRMED for a model that never recalls at the reference noise. Validity V checks sigma = 0 only. A tightening before the tag.
- **D20.** Admission and source binding: analyze.py resolves the PREREG tag with git, refuses unless prereg.json is the tagged one, and admits manifests only as stated under the flag 'admission', every manifest and the gate naming exactly the commit the tag resolves to. V's gate counts only if tests passed, the PREREG check passed, the mutant runner exited 0 and every registered mutant of mutants.json was killed in that same gate. tools/confirm.py deletes any earlier mutant status before a gate, holds one lock across gate, run and rerun, refuses existing manifests that analyze.py would not admit, and in every worker checks the tracked experiment files against the tag and then imports phs/ from bytes read once and checked against the tag's git blobs (the code that runs is the code that was checked). *Why:* Review rounds 1-2 (Codex): a hash claim alone admitted manifests with foreign cells, n = 1 or another configuration; duplicates selected results by file-name order; a stale or partial mutant matrix could satisfy V; consistency between claims without resolving the tag admitted manifests and a gate with no or a foreign tag commit; a worker could import a source changed after its check. Residual: tools/confirm.py itself is read when a worker starts; an agent with write access to the checkout during a run is outside what the runner defends.
- **D21.** Non-finite outputs: in evaluation a sequence with any non-finite logit counts as wrong and is counted in the cell's 'nonfinite'; loss, R_slip and erank average the finite sequences (R_slip and erank are null for a cell without one, and analyze.py averages the cells that have them). A validation checkpoint is eligible only with every output finite and a finite loss. *Why:* Review rounds 1-2 (Codex): an argmax over NaN logits returned an apparent chance accuracy (0.143) with NaN loss; finite but extreme logits gave an infinite validation loss that stayed eligible; a cell with no finite sequence crashed the aggregation.

## Calibration

- Rounds 1-5 (calibration/rounds/, calibration/SUMMARY.md): 12 configurations for Pure-PHS, 12 for Soliton-PHS, 10 for Control-GRU, 12 for Control-GRU-pm, each trained on tasks A and B at T = 32, seed 100, 4000 steps.
- Selected: Pure p07 and Soliton s07 (N, d = 32, 2; dt 1; S 4; gamma 0.5; u_max 2; lr 1e-2; Soliton m = 2), Control-GRU g06 (lr 7e-3), Control-GRU-pm q07 (lr 3e-2).
- Both PHS models stayed at chance on task B in every configuration except Soliton s07 (validation 0.51, rising at step 4000); Control-GRU-pm learned task B but task A only at lr >= 3e-2.
- Probe sets 1-2 (calibration/probes/): T = 32 at 12000 steps, T = 128 at 4000 steps and T = 16 at 12000 steps on sigma up to 1.0 or 2.0; they set the evaluation grid (D14) and the step count (D15).
- Calibration rounds ran from commits f246d4b, 5eb407a, bc2b977, 59b216c and f6f3b2b; probes from 003a360 and 51dc475; each record carries its commit (round 1 by this note).

## Execution

- Command: `make prereg-check && .venv/bin/python tools/confirm.py gate && .venv/bin/python tools/confirm.py run --workers 12` from phs-soliton/ at the tag; then `make analyze`.
- Runner refusals: no tag; prereg.json differs from the tagged one; any tracked experiment file outside results/, RESULTS.md, REPORT.md, figures/ differs from the tag or is uncommitted; a manifest names another PREREG; no gate for this PREREG.
- Parallelism: 12 single-threaded runs. Peak RSS of one run measured at 1.36 GB (T = 128, (32, 2)), so min(12, floor(24 GB / 1.36 GB)) = 12.
- Order: longest first (PHS before GRU, larger T first). Every run writes results/manifests/<task>-<kind>-T<T>-s<seed>.json: commit, tag, PREREG hash, versions, machine, seed, configuration, wall time, peak RSS, learned physics, flags, validation history, metrics per cell. The runner resumes by skipping existing manifests.
- Time estimate from the probes under a 12-process load: PHS (32, 2) about 0.11, 0.21, 0.42 and 0.83 s per step at T = 16, 32, 64, 128, i.e. about 1.6 s per step per (PHS model, task, seed); 20 such sets x 24000 steps = 7.5e5 worker-seconds, plus about 2.5e4 for the 41 test cells per run and about 1e4 for the GRU runs; over 12 workers about 18-20 h. Under the 48 h limit.

## Review

- Round 1: Codex (codex-cli 0.164.0, read-only sandbox), blind prompt (what and where, no conclusions), on commit cbc355f. AMEND: 4 P1 + 3 P2. Text: reviews/prereg-v1-codex-r1.md; prompt reviews/prereg-v1-codex-prompt.txt.
- Round 2, scoped to the seven findings, on commit 8d70cfd. AMEND: 4 closed (room, stale gate, duplicates, lambda floor); open P1 tag correspondence (claims agreed without resolving the tag), P2 check-then-import window in workers, P2 infinite validation loss still eligible; new P2 crash on a cell without finite sequences. Text: reviews/prereg-v1-codex-r2.md.
- All round-2 items fixed with tests and mutants (D20, D21). Round 3 below.
- Round 1 found no test-cell leakage in the 92 calibration jobs and 18 probes (all seed 100, calib digests).

## Frozen files

- `analyze.py`: `227f952d175fbee3d1c784f5d8cfd3bf37f6c05a9d3b79443143a1e9734be98b`
