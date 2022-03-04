#include <mpi.h>
#include <cstdio>
#include <random>
#include <cassert>
#include <span>

constexpr auto VECTOR_SIZE = 50;
constexpr auto STEP_SIZE = 9;

constexpr auto DATA_INT_COUNT = 2;

enum Status {
	KeepGoing,
	Done,
};

struct Slice {
	int start_index;
	int len;
};

constexpr auto DONE_MSG = Slice{-1, -1};

auto is_done(Slice slice) -> bool { return slice.start_index == DONE_MSG.start_index && slice.len == DONE_MSG.len; }

auto advance(int* index, bool* done) -> Slice {
	const auto old_index = *index;
	const auto len = std::min(VECTOR_SIZE - *index, STEP_SIZE);
	const auto result = Slice{*index, len};
	*index += len;
	*done = old_index == *index;
	return result;
}

auto generate_vector(std::mt19937& generator, int n) -> std::vector<float> {
	auto result = std::vector<float>(n);
	auto dist = std::uniform_real_distribution<float>{-100, 100};

	for (auto& x : result) {
		x = dist(generator);
	}

	return result;
}

void send_to(int destination, Slice data) {
	MPI_Send(&data, DATA_INT_COUNT, MPI_INT, destination, destination, MPI_COMM_WORLD);
}

auto receive_from(int source) -> Slice {
	auto result = Slice{};
	MPI_Recv(&result, DATA_INT_COUNT, MPI_INT, source, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	return result;
}

void send_result_to(int destination, float result) {
	MPI_Send(&result, 1, MPI_FLOAT, destination, destination, MPI_COMM_WORLD);
}

auto receive_result(int* source) -> float {
	auto result = float{};
	auto status = MPI_Status{};
	MPI_Recv(&result, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	*source = status.MPI_SOURCE;
	return result;
}

auto send_work_to(int worker, int* index) -> Status {
	auto done = bool{};
	const auto data = advance(index, &done);

	if (done) {
		return Done;
	}

	send_to(worker, data);
	return KeepGoing;
}

void send_initial_work(int* index, int num_workers) {
	for (auto worker = 0; worker < num_workers; ++worker) {
		if (send_work_to(worker, index) == Done) {
			return;
		}
	}
}

auto accumulate_and_reschedule(int* index, int num_workers) -> float {
	auto num_finished_workers = 0;
	auto result = 0.f;
	for (;;) {
		auto finished_worker = int{};
		result += receive_result(&finished_worker);
		if (send_work_to(finished_worker, index) == Done) {
			if (++num_finished_workers == num_workers) {
				return result;
			}
		}
	}
}

void signal_done(int num_workers) {
	for (auto worker = 0; worker < num_workers; ++worker) {
		send_to(worker, DONE_MSG);
	}
}

auto dot_product(const float* a, const float* b, int n) -> float {
	auto result = 0.f;

	for (auto i = 0; i < n; ++i) {
		result += a[i] * b[i];
	}

	return result;
}

auto run_master(int num_workers) -> float {
	auto state = 0;

	send_initial_work(&state, num_workers);
	const auto result = accumulate_and_reschedule(&state, num_workers);
	signal_done(num_workers);

	return result;
}

void run_worker(const std::vector<float>& a, const std::vector<float>& b, int master) {
	for (;;) {
		const auto data = receive_from(master);

		if (is_done(data)) {
			return;
		}

		const auto result = dot_product(&a[data.start_index], &b[data.start_index], data.len);
		send_result_to(master, result);
	}
}

auto run_reference(const std::vector<float>& a, const std::vector<float>& b) -> float {
	return dot_product(a.data(), b.data(), VECTOR_SIZE);
}

auto main(int argc, char** argv) -> int {
	MPI_Init(&argc, &argv);

	auto rank = int{}, size = int{};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	assert(size >= 2 && "Need at least one master and one worker process");

	auto device = std::random_device{};
	auto generator = std::mt19937{device()};


	auto a = std::vector<float>{}, b = std::vector<float>{};
	a = generate_vector(generator, VECTOR_SIZE);
	b = generate_vector(generator, VECTOR_SIZE);

	const auto master = size - 1;
	const auto num_workers = size - 1; // coincidentally same definition but different semantics

	if (rank == master) {

		printf("master thread has rank %d\n", master);

		const auto result = run_master(num_workers);
		const auto reference = run_reference(a, b);

		printf("calculated dot product on %d element vectors: %f (expected %f)\n", VECTOR_SIZE, result, reference);
	} else {
		run_worker(a, b, master);
	}

	MPI_Finalize();
}
