#include <mpi.h>
#include <span>
#include <utility>
#include <climits>

template<typename E> struct Type {};
template<> struct Type<int> { constexpr static auto datatype = MPI_INT; };
template<> struct Type<float> { constexpr static auto datatype = MPI_FLOAT; };
template<> struct Type<double> { constexpr static auto datatype = MPI_DOUBLE; };
template<> struct Type<std::pair<int, int>> { constexpr static auto datatype = MPI_2INT; };


namespace mpi
{

using Two_Ints = std::pair<int, int>;

struct Tag {
	int raw = 0;
	Tag() = default;
	explicit constexpr Tag(int value) : raw{value} {};
	friend auto operator<=>(const Tag&, const Tag&) = default;
	friend auto operator<=>(const Tag& a, const int& b) { return a.raw <=> b; };
};

constexpr auto USER_TAG(int i = 0) -> Tag { return Tag{INT_MAX - i}; }

struct Rank {
	int raw = 0;
	Rank() = default;
	explicit constexpr Rank(int value) : raw{value} {};
	friend auto operator<=>(const Rank&, const Rank&) = default;
	friend auto operator<=>(const Rank& a, const int& b) { return a.raw <=> b; };
	friend auto operator++(Rank& a) { return ++a.raw; }
	auto as_tag() -> Tag { return Tag{raw}; }
};

struct Status {
	Rank source;
	Tag tag;
};

constexpr auto ANY_SOURCE = Rank{MPI_ANY_SOURCE};
constexpr auto ANY_TAG = Tag{MPI_ANY_TAG};

auto rank() -> Rank {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return Rank{rank};
}

auto num_ranks() -> int {
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;
}

template<typename E>
void send(std::span<E> elements, Rank dest, Tag tag) {
	MPI_Send(elements.data(), elements.size(), Type<E>::datatype, dest.raw, tag.raw, MPI_COMM_WORLD);
}

template<typename T>
void send(const T& element, Rank dest, Tag tag) {
	MPI_Send(&element, 1, Type<T>::datatype, dest.raw, tag.raw, MPI_COMM_WORLD);
}

void send(Rank dest, Tag tag) {
	MPI_Send(nullptr, 0, MPI_BYTE, dest.raw, tag.raw, MPI_COMM_WORLD);
}

template<typename T>
auto receive(Rank source, Tag tag, Status* status = nullptr) -> T {
	T result;
	auto raw_status = MPI_Status{};
	MPI_Recv(&result, 1, Type<T>::datatype, source.raw, tag.raw, MPI_COMM_WORLD, &raw_status);

	if (status) {
		status->source = Rank{raw_status.MPI_SOURCE};
		status->tag = Tag{raw_status.MPI_TAG};
	}

	return result;
}

struct Init_Guard {
	Init_Guard(int& argc, char**& argv) {
		MPI_Init(&argc, &argv);
	}

	~Init_Guard() {
		MPI_Finalize();
	}
};

}
