#include "mpi.hpp"
#include <cstdio>
#include <random>
#include <cassert>

constexpr auto VECTOR_SIZE = 50;
constexpr auto STEP_SIZE = 9;

enum Status {
	KeepGoing,
	Done,
};

auto starting_index(const mpi::Two_Ints& pair) -> int { return pair.first; };
auto span_length(const mpi::Two_Ints& pair) -> int { return pair.second; };

struct Input {
	float a[VECTOR_SIZE];
	float b[VECTOR_SIZE];
};

void fill_vector(std::mt19937& generator, float* vector) {
	auto dist = std::uniform_real_distribution<float>{-100, 100};

	for (auto i = 0; i < VECTOR_SIZE; ++i) {
		vector[i] = dist(generator);
	}
}

auto generate_input() -> Input {
	auto device = std::random_device{};
	auto generator = std::mt19937{device()};
	auto input = Input{};
	fill_vector(generator, input.a);
	fill_vector(generator, input.b);
	return input;
}

constexpr auto DONE_TAG = mpi::USER_TAG();

auto advance(int* index, bool* done) -> mpi::Two_Ints {
	const auto old_index = *index;
	const auto len = std::min(VECTOR_SIZE - *index, STEP_SIZE);
	const auto result = mpi::Two_Ints{*index, len};
	*index += len;
	*done = old_index == *index;
	return result;
}

void broadcast_input_from(mpi::Rank source, Input& input) {
	MPI_Bcast(&input, VECTOR_SIZE * 2, MPI_FLOAT, source.raw, MPI_COMM_WORLD);
}

inline auto receive_broadcast_input_from(mpi::Rank source) -> Input {
	auto input = Input{};
	MPI_Bcast(&input, VECTOR_SIZE * 2, MPI_FLOAT, source.raw, MPI_COMM_WORLD);
	return input;
}

void send_to(mpi::Rank destination, mpi::Two_Ints data) {
	mpi::send(data, destination, destination.as_tag());
}

void send_result_to(mpi::Rank destination, float result) {
	mpi::send(result, destination, destination.as_tag());
}

auto receive_result(mpi::Rank* source) -> float {
	auto status = mpi::Status{};
	const auto result = mpi::receive<float>(mpi::ANY_SOURCE, mpi::ANY_TAG, &status);
	*source = status.source;
	return result;
}

auto send_work_to(mpi::Rank worker, int* index) -> Status {
	auto done = bool{};
	const auto data = advance(index, &done);

	if (done) {
		return Done;
	}

	send_to(worker, data);
	return KeepGoing;
}

void send_initial_work(int* index, int num_workers) {
	for (auto worker = mpi::Rank{0}; worker < num_workers; ++worker) {
		if (send_work_to(worker, index) == Done) {
			return;
		}
	}
}

auto accumulate_and_reschedule(int* index, int num_workers) -> float {
	auto num_finished_workers = 0;
	auto result = 0.f;
	for (;;) {
		auto finished_worker = mpi::Rank{};
		result += receive_result(&finished_worker);
		if (send_work_to(finished_worker, index) == Done) {
			if (++num_finished_workers == num_workers) {
				return result;
			}
		}
	}
}

void signal_done(int num_workers) {
	for (auto worker = mpi::Rank{0}; worker < num_workers; ++worker) {
		send(worker, DONE_TAG);
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

void run_worker(const Input& input, mpi::Rank master) {
	for (;;) {
		auto status = mpi::Status{};
		const auto data = mpi::receive<mpi::Two_Ints>(master, mpi::ANY_TAG, &status);

		if (status.tag == DONE_TAG) {
			return;
		}

		const auto result = dot_product(&input.a[starting_index(data)], &input.b[starting_index(data)], span_length(data));
		send_result_to(master, result);
	}
}

auto run_reference(const Input& input) -> float {
	return dot_product(input.a, input.b, VECTOR_SIZE);
}

auto main(int argc, char** argv) -> int {
	const auto guard = mpi::Init_Guard{argc, argv};

	const auto rank = mpi::rank();
	const auto size = mpi::num_ranks();

	assert(size >= 2 && "Need at least one master and one worker process");

	const auto master = mpi::Rank{size - 1};
	const auto num_workers = size - 1;

	if (rank == master) {
		auto input = generate_input();
		broadcast_input_from(master, input);

		printf("master thread has rank %d\n", master.raw);

		const auto result = run_master(num_workers);
		const auto reference = run_reference(input);

		printf("calculated dot product on %d element vectors: %f (expected %f)\n", VECTOR_SIZE, result, reference);
	} else {
		const auto input = receive_broadcast_input_from(master);

		run_worker(input, master);
	}
}
